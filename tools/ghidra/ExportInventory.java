// Headless post-script: writes the analyzed program's functions and defined
// strings as TSV files into the folder given as the first script argument.
//   <program>.functions.tsv  entry, body size, name
//   <program>.strings.tsv    address, length, value (tabs/newlines escaped)
// @category Gyroship
import ghidra.app.script.GhidraScript;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

public class ExportInventory extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: ExportInventory.java <output folder>");
        }
        File out = new File(args[0]);
        out.mkdirs();
        String name = currentProgram.getName();

        int functions = 0;
        try (PrintWriter w = new PrintWriter(new File(out, name + ".functions.tsv"), StandardCharsets.UTF_8)) {
            for (Function f : currentProgram.getFunctionManager().getFunctions(true)) {
                w.printf("%s\t%d\t%s%n", f.getEntryPoint(), f.getBody().getNumAddresses(), f.getName(true));
                functions++;
            }
        }
        int strings = 0;
        try (PrintWriter w = new PrintWriter(new File(out, name + ".strings.tsv"), StandardCharsets.UTF_8)) {
            for (Data d : currentProgram.getListing().getDefinedData(true)) {
                if (monitor.isCancelled()) {
                    break;
                }
                if (!StringDataInstance.isString(d)) {
                    continue;
                }
                String value = StringDataInstance.getStringDataInstance(d).getStringValue();
                if (value == null) {
                    continue;
                }
                value = value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r");
                w.printf("%s\t%d\t%s%n", d.getAddress(), d.getLength(), value);
                strings++;
            }
        }
        println(name + ": " + functions + " functions, " + strings + " strings exported to " + out);
    }
}
