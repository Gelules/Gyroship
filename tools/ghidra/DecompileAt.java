// Headless post-script: decompiles the functions that contain the given
// addresses, and the functions they call directly, one level down.
//
// Arguments: <output folder> <hex address>...
// Writes <output>/<entry>.c for each function. Decompiler output is derived
// from the game: keep it out of the repository.
// @category Gyroship
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashSet;
import java.util.Set;

public class DecompileAt extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            throw new IllegalArgumentException("usage: DecompileAt.java <output folder> <hex address>...");
        }
        File out = new File(args[0]);
        out.mkdirs();
        Set<Function> todo = new LinkedHashSet<>();
        for (int i = 1; i < args.length; i++) {
            Function f = getFunctionContaining(toAddr(Long.parseUnsignedLong(args[i].replaceFirst("^0x", ""), 16)));
            if (f == null) {
                println("no function at " + args[i]);
                continue;
            }
            todo.add(f);
            todo.addAll(f.getCalledFunctions(monitor));
        }
        DecompInterface dec = new DecompInterface();
        dec.openProgram(currentProgram);
        for (Function f : todo) {
            DecompileResults res = dec.decompileFunction(f, 120, monitor);
            String c = res.decompileCompleted() ? res.getDecompiledFunction().getC() : "/* decompilation failed: " + res.getErrorMessage() + " */";
            try (PrintWriter w = new PrintWriter(new File(out, f.getEntryPoint() + ".c"), StandardCharsets.UTF_8)) {
                w.println("// " + f.getName(true) + " at " + f.getEntryPoint());
                w.print(c);
            }
        }
        dec.dispose();
        println(todo.size() + " functions decompiled into " + out);
    }
}
