// Headless post-script: finds the functions that refer to strings matching a
// regular expression, directly from code or through a table entry that
// pairs a string pointer with a function pointer (script bindings), and
// decompiles each of them.
//
// Arguments: <regex> <output folder>; the match ignores case (no (?i) needed,
// which the analyzeHeadless batch file would not pass through).
// Writes <output>/refs.tsv (string address, value, kind, function entry,
// function name) and <output>/<entry>.c for every function found.
// Decompiler output is derived from the game: keep it out of the repository.
// @category Gyroship
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.symbol.Reference;
import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

public class FindByStrings extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: FindByStrings.java <regex> <output folder>");
        }
        Pattern pattern = Pattern.compile(args[0], Pattern.CASE_INSENSITIVE);
        File out = new File(args[1]);
        out.mkdirs();
        Memory mem = currentProgram.getMemory();
        Map<Function, String> found = new LinkedHashMap<>();

        try (PrintWriter w = new PrintWriter(new File(out, "refs.tsv"), StandardCharsets.UTF_8)) {
            for (Data d : currentProgram.getListing().getDefinedData(true)) {
                if (monitor.isCancelled()) {
                    break;
                }
                if (!StringDataInstance.isString(d)) {
                    continue;
                }
                String value = StringDataInstance.getStringDataInstance(d).getStringValue();
                if (value == null || !pattern.matcher(value).find()) {
                    continue;
                }
                for (Reference r : getReferencesTo(d.getAddress())) {
                    Address from = r.getFromAddress();
                    Function f = getFunctionContaining(from);
                    if (f != null) {
                        record(w, found, d, value, "code", f);
                        continue;
                    }
                    // A data slot pointing at the string: look for a code pointer next to it.
                    for (int delta : new int[] {8, -8, 16}) {
                        try {
                            long p = mem.getLong(from.add(delta));
                            Address target = toAddr(p);
                            Function g = target == null ? null : getFunctionAt(target);
                            if (g != null) {
                                record(w, found, d, value, "table" + (delta > 0 ? "+" : "") + delta, g);
                                break;
                            }
                        } catch (Exception e) {
                            // not readable memory: no neighbour there
                        }
                    }
                }
            }
        }

        DecompInterface dec = new DecompInterface();
        dec.openProgram(currentProgram);
        int done = 0;
        for (Map.Entry<Function, String> e : found.entrySet()) {
            if (monitor.isCancelled()) {
                break;
            }
            Function f = e.getKey();
            DecompileResults res = dec.decompileFunction(f, 120, monitor);
            String c = res.decompileCompleted() ? res.getDecompiledFunction().getC() : "/* decompilation failed: " + res.getErrorMessage() + " */";
            try (PrintWriter w = new PrintWriter(new File(out, f.getEntryPoint() + ".c"), StandardCharsets.UTF_8)) {
                w.println("// " + f.getName(true) + " at " + f.getEntryPoint() + " — refers to: " + e.getValue());
                w.print(c);
            }
            done++;
        }
        dec.dispose();
        println(found.size() + " functions refer to /" + args[0] + "/, " + done + " decompiled into " + out);
    }

    private void record(PrintWriter w, Map<Function, String> found, Data d, String value, String kind, Function f) {
        w.printf("%s\t%s\t%s\t%s\t%s%n", d.getAddress(), value.replace("\t", "\\t"), kind, f.getEntryPoint(), f.getName(true));
        found.merge(f, value, (a, b) -> a.contains(b) ? a : a + " | " + b);
    }
}
