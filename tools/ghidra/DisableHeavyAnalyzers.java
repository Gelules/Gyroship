// Headless pre-script: turns off auto-analysis passes that cost the most memory
// and time on very large modules. Jump tables are recovered by gyroship-analysis
// instead, so the inventory does not need them.
// @category Gyroship
import ghidra.app.script.GhidraScript;

public class DisableHeavyAnalyzers extends GhidraScript {
    private static final String[] HEAVY = {
        "Decompiler Switch Analysis",
        "Decompiler Parameter ID",
        "Call Convention ID",
    };

    @Override
    protected void run() throws Exception {
        for (String name : HEAVY) {
            setAnalysisOption(currentProgram, name, "false");
            println("disabled analyzer: " + name);
        }
    }
}
