// Headless post-script: disassembles and creates a function at each given
// entry address that no function contains yet, so that the scripts run after
// it (DecompileAt, DecompileCallers) reach code the auto-analysis missed.
// Run it with -readOnly to leave the project unchanged.
//
// Arguments: <hex address>...
// @category Gyroship
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public class MakeFunctions extends GhidraScript {
    @Override
    protected void run() throws Exception {
        for (String arg : getScriptArgs()) {
            Address a = toAddr(Long.parseUnsignedLong(arg.replaceFirst("^0x", ""), 16));
            if (getFunctionContaining(a) != null) {
                continue;
            }
            disassemble(a);
            Function f = createFunction(a, null);
            println(f == null ? "could not create a function at " + a : "created " + f.getName() + ", " + f.getBody().getNumAddresses() + " bytes");
        }
    }
}
