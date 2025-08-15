import java.io.*;
import java.nio.file.*;

/**
 * Extension methods for ComprehensiveFrameworkIntegration class
 * These methods should be added to the main ComprehensiveFrameworkIntegration class
 */
public class ComprehensiveFrameworkExtensions {
    
    /**
     * Export comprehensive results - add this method to ComprehensiveFrameworkIntegration
     */
    public void exportComprehensiveResults(
            FrameworkAnalysisResult frameworkResult,
            List<CollaborationMatch> collaborationMatches,
            ValidationResult lstmValidation,
            IntegratedAnalysisResult integratedResult) throws IOException {
        
        ComprehensiveFrameworkMethods.exportComprehensiveResults(
            frameworkResult, collaborationMatches, lstmValidation, 
            integratedResult, outputDirectory);
    }
    
    /**
     * Generate theoretical insights - add this method to ComprehensiveFrameworkIntegration
     */
    public void generateTheoreticalInsights(IntegratedAnalysisResult integratedResult) throws IOException {
        ComprehensiveFrameworkMethods.generateTheoreticalInsights(integratedResult, outputDirectory);
        
        System.out.println("   Generated comprehensive theoretical analysis:");
        System.out.println("   • Framework performance analysis");
        System.out.println("   • Theoretical consistency validation");
        System.out.println("   • Emergent behavior identification");
        System.out.println("   • Research recommendations");
        System.out.println("   • Limitations and future work discussion");
    }
}
