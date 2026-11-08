// Headless post-script: lists the callers of the given functions, one level
// up, and decompiles them. A caller is a function holding a reference (call,
// branch or pointer) to the function's entry or to one of its thunks (PLT
// stubs, branch veneers), so calls through those are found too.
//
// Arguments: <output folder> <max callers> <target>...
// A target is a 0x-prefixed address (the function containing it) or the name
// of an external function, mangled or not (every import with that name).
// Writes <output>/callers.tsv (target entry, target name, reference address,
// reference type, caller entry, caller name; "-" for a reference from outside
// any function) and <output>/<entry>.c for each caller of a target that has at
// most <max callers> callers; a target with more is only listed.
// Decompiler output is derived from the game: keep it out of the repository.
// @category Gyroship
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.ExternalLocation;
import ghidra.program.model.symbol.Reference;
import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DecompileCallers extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 3) {
            throw new IllegalArgumentException("usage: DecompileCallers.java <output folder> <max callers> <target>...");
        }
        File out = new File(args[0]);
        out.mkdirs();
        int max = Integer.parseInt(args[1]);
        Map<Function, String> todo = new LinkedHashMap<>();

        try (PrintWriter w = new PrintWriter(new File(out, "callers.tsv"), StandardCharsets.UTF_8)) {
            for (int i = 2; i < args.length; i++) {
                List<Function> targets = resolve(args[i]);
                if (targets.isEmpty()) {
                    println("no function for " + args[i]);
                }
                for (Function t : targets) {
                    Set<Address> entries = new LinkedHashSet<>();
                    entries.add(t.getEntryPoint());
                    Address[] thunks = t.getFunctionThunkAddresses(true);
                    if (thunks != null) {
                        entries.addAll(Arrays.asList(thunks));
                    }
                    Map<Function, String> callers = new LinkedHashMap<>();
                    for (Address e : entries) {
                        for (Reference r : getReferencesTo(e)) {
                            Address from = r.getFromAddress();
                            Function c = getFunctionContaining(from);
                            if (c != null && entries.contains(c.getEntryPoint())) {
                                continue; // a thunk's own branch to the next hop
                            }
                            w.printf("%s\t%s\t%s\t%s\t%s\t%s%n", t.getEntryPoint(), t.getName(true), from, r.getReferenceType(),
                                c == null ? "-" : c.getEntryPoint(), c == null ? "-" : c.getName(true));
                            if (c != null) {
                                callers.merge(c, t.getName(true) + " from " + from, (a, b) -> a + ", " + from);
                            }
                        }
                    }
                    if (callers.size() > max) {
                        println(args[i] + ": " + callers.size() + " callers, listed only");
                        continue;
                    }
                    callers.forEach((c, note) -> todo.merge(c, note, (a, b) -> a + "; " + b));
                }
            }
        }

        DecompInterface dec = new DecompInterface();
        dec.openProgram(currentProgram);
        for (Map.Entry<Function, String> e : todo.entrySet()) {
            if (monitor.isCancelled()) {
                break;
            }
            Function f = e.getKey();
            DecompileResults res = dec.decompileFunction(f, 120, monitor);
            String c = res.decompileCompleted() ? res.getDecompiledFunction().getC() : "/* decompilation failed: " + res.getErrorMessage() + " */";
            try (PrintWriter w = new PrintWriter(new File(out, f.getEntryPoint() + ".c"), StandardCharsets.UTF_8)) {
                w.println("// " + f.getName(true) + " at " + f.getEntryPoint() + " — calls " + e.getValue());
                w.print(c);
            }
        }
        dec.dispose();
        println(todo.size() + " callers decompiled into " + out);
    }

    private List<Function> resolve(String arg) {
        List<Function> found = new ArrayList<>();
        if (arg.startsWith("0x")) {
            Function f = getFunctionContaining(toAddr(Long.parseUnsignedLong(arg.substring(2), 16)));
            if (f != null) {
                found.add(f);
            }
            return found;
        }
        for (Function f : currentProgram.getFunctionManager().getExternalFunctions()) {
            ExternalLocation loc = f.getExternalLocation();
            if (arg.equals(f.getName()) || arg.equals(f.getName(true)) || arg.equals(loc.getLabel())
                || arg.equals(loc.getOriginalImportedName())) {
                found.add(f);
            }
        }
        return found;
    }
}
