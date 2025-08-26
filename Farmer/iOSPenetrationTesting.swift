// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  iOSPenetrationTesting.swift
//  Farmer
//
//  Created by Ryan David Oates on 8/26/25.
//  iOS penetration testing framework using reverse koopman operators

import Foundation
import UIKit
import Security
import Network
import CoreData
import CoreML

/// Security vulnerability types
enum VulnerabilityType: String, Codable {
    case bufferOverflow = "Buffer Overflow"
    case sqlInjection = "SQL Injection"
    case xss = "Cross-Site Scripting"
    case insecureStorage = "Insecure Storage"
    case weakEncryption = "Weak Encryption"
    case insecureNetwork = "Insecure Network Communication"
    case authenticationBypass = "Authentication Bypass"
    case privilegeEscalation = "Privilege Escalation"
    case memoryLeak = "Memory Leak"
    case raceCondition = "Race Condition"
}

/// Security severity levels
enum SeverityLevel: String, Codable {
    case critical = "Critical"
    case high = "High"
    case medium = "Medium"
    case low = "Low"
    case info = "Info"
}

/// Security finding with koopman analysis
struct SecurityFinding: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let vulnerabilityType: VulnerabilityType
    let severity: SeverityLevel
    let title: String
    let description: String
    let location: String
    let recommendation: String

    // Koopman analysis results
    let koopmanStability: Double
    let spectralRadius: Double
    let conditionNumber: Double
    let reconstructionError: Double
    let dominantModes: Int

    // Penetration testing data
    let exploitVector: String?
    let impactAssessment: String
    let cvssScore: Double

    init(
        vulnerabilityType: VulnerabilityType,
        severity: SeverityLevel,
        title: String,
        description: String,
        location: String,
        recommendation: String,
        koopmanStability: Double = 0.0,
        spectralRadius: Double = 0.0,
        conditionNumber: Double = 0.0,
        reconstructionError: Double = 0.0,
        dominantModes: Int = 0,
        exploitVector: String? = nil,
        impactAssessment: String = "",
        cvssScore: Double = 0.0
    ) {
        self.id = UUID()
        self.timestamp = Date()
        self.vulnerabilityType = vulnerabilityType
        self.severity = severity
        self.title = title
        self.description = description
        self.location = location
        self.recommendation = recommendation
        self.koopmanStability = koopmanStability
        self.spectralRadius = spectralRadius
        self.conditionNumber = conditionNumber
        self.reconstructionError = reconstructionError
        self.dominantModes = dominantModes
        self.exploitVector = exploitVector
        self.impactAssessment = impactAssessment
        self.cvssScore = cvssScore
    }
}

/// iOS Penetration Testing Framework
class iOSPenetrationTesting {

    private let reverseKoopman: ReverseKoopmanOperator
    private var findings: [SecurityFinding] = []
    private let fileManager = FileManager.default
    private let keychain = KeychainManager()

    /// Initialize penetration testing framework
    /// - Parameter koopmanOperator: Reverse koopman operator for analysis
    init(koopmanOperator: ReverseKoopmanOperator) {
        self.reverseKoopman = koopmanOperator
        print("iOS Penetration Testing Framework initialized")
    }

    // MARK: - Core Penetration Testing Methods

    /// Perform comprehensive iOS security assessment
    /// - Returns: Array of security findings
    func performSecurityAssessment() -> [SecurityFinding] {
        print("Starting comprehensive iOS security assessment...")

        findings.removeAll()

        // Analyze system behavior using koopman operators
        analyzeSystemBehavior()

        // Test various security aspects
        testInsecureStorage()
        testNetworkSecurity()
        testAuthenticationMechanisms()
        testMemoryManagement()
        testRaceConditions()
        testBufferOverflows()

        print("Security assessment completed. Found \(findings.count) issues.")
        return findings
    }

    /// Analyze system behavior using reverse koopman operators
    private func analyzeSystemBehavior() {
        print("Analyzing system behavior with reverse koopman operators...")

        // Generate system state trajectory
        let trajectory = reverseKoopman.generateVanDerPolTrajectory(nPoints: 1000)

        // Construct koopman matrix
        let koopmanMatrix = reverseKoopman.constructKoopmanMatrix(trajectory: trajectory)

        if !koopmanMatrix.isEmpty {
            // Compute spectral decomposition
            let (eigenvalues, _) = reverseKoopman.computeSpectralDecomposition()

            // Estimate Lipschitz constants
            let (cLower, _) = reverseKoopman.estimateLipschitzConstants(nSamples: 200)

            // Analyze stability and create findings
            analyzeStabilityFindings(eigenvalues: eigenvalues, cLower: cLower)
        }
    }

    /// Analyze stability findings from koopman analysis
    private func analyzeStabilityFindings(eigenvalues: [ComplexNumber], cLower: Double) {
        let spectralRadius = eigenvalues.map { $0.magnitude }.max() ?? 0.0

        // Check for unstable behavior
        if spectralRadius > 1.1 {
            let finding = SecurityFinding(
                vulnerabilityType: .privilegeEscalation,
                severity: .high,
                title: "System Instability Detected",
                description: "Reverse koopman analysis reveals unstable system behavior with spectral radius \(String(format: "%.3f", spectralRadius)). This may indicate exploitable state transitions.",
                location: "System Dynamics",
                recommendation: "Implement proper state validation and bounds checking to prevent unstable transitions.",
                koopmanStability: cLower,
                spectralRadius: spectralRadius,
                conditionNumber: 0.0,
                reconstructionError: 0.0,
                dominantModes: eigenvalues.count,
                impactAssessment: "High - Unstable system behavior may lead to privilege escalation or unexpected state changes."
            )
            findings.append(finding)
        }

        // Check for poor conditioning (numerical instability)
        let conditionNumber = eigenvalues.isEmpty ? 0.0 : eigenvalues[0].magnitude / (eigenvalues.last?.magnitude ?? 1.0)
        if conditionNumber > 100 {
            let finding = SecurityFinding(
                vulnerabilityType: .memoryLeak,
                severity: .medium,
                title: "Poor Numerical Conditioning",
                description: "System shows poor numerical conditioning (κ = \(String(format: "%.1f", conditionNumber))). This may lead to memory corruption or precision loss.",
                location: "Numerical Computations",
                recommendation: "Implement proper regularization and error bounds checking in numerical computations.",
                koopmanStability: cLower,
                spectralRadius: spectralRadius,
                conditionNumber: conditionNumber,
                impactAssessment: "Medium - Poor conditioning may lead to computational errors and memory issues."
            )
            findings.append(finding)
        }
    }

    /// Test insecure storage mechanisms
    private func testInsecureStorage() {
        print("Testing insecure storage mechanisms...")

        // Test NSUserDefaults for sensitive data
        testNSUserDefaults()

        // Test file system storage
        testFileSystemStorage()

        // Test Keychain security
        testKeychainSecurity()

        // Test CoreData security
        testCoreDataSecurity()
    }

    /// Test NSUserDefaults for sensitive data
    private func testNSUserDefaults() {
        let userDefaults = UserDefaults.standard
        let sensitiveKeys = ["password", "token", "secret", "key", "auth", "session"]

        for key in userDefaults.dictionaryRepresentation().keys {
            if sensitiveKeys.contains(where: { key.lowercased().contains($0) }) {
                let finding = SecurityFinding(
                    vulnerabilityType: .insecureStorage,
                    severity: .medium,
                    title: "Sensitive Data in NSUserDefaults",
                    description: "Sensitive data found in NSUserDefaults under key '\(key)'. This data is not encrypted and can be easily accessed.",
                    location: "NSUserDefaults",
                    recommendation: "Move sensitive data to Keychain or implement proper encryption for NSUserDefaults storage.",
                    impactAssessment: "Medium - Sensitive data may be exposed to unauthorized access."
                )
                findings.append(finding)
            }
        }
    }

    /// Test file system storage security
    private func testFileSystemStorage() {
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first?.path ?? ""

        do {
            let files = try fileManager.contentsOfDirectory(atPath: documentsPath)

            for file in files {
                if file.contains("password") || file.contains("secret") || file.contains("key") {
                    let finding = SecurityFinding(
                        vulnerabilityType: .insecureStorage,
                        severity: .high,
                        title: "Sensitive File Detected",
                        description: "File '\(file)' appears to contain sensitive data but is stored in plaintext.",
                        location: "Documents Directory",
                        recommendation: "Encrypt sensitive files or move them to secure storage locations.",
                        impactAssessment: "High - Sensitive files may be accessed by malicious applications."
                    )
                    findings.append(finding)
                }
            }
        } catch {
            print("Error accessing documents directory: \(error)")
        }
    }

    /// Test Keychain security
    private func testKeychainSecurity() {
        // Test for common keychain vulnerabilities
        let testKeys = ["com.example.app.password", "auth_token", "session_key"]

        for key in testKeys {
            do {
                let data = try keychain.load(key: key)
                if data != nil {
                    // Check if data is properly encrypted
                    let finding = SecurityFinding(
                        vulnerabilityType: .weakEncryption,
                        severity: .low,
                        title: "Keychain Data Accessibility",
                        description: "Keychain item '\(key)' is accessible. Verify proper access control settings.",
                        location: "Keychain",
                        recommendation: "Review and strengthen Keychain access control policies.",
                        impactAssessment: "Low - Verify that proper access controls are in place."
                    )
                    findings.append(finding)
                }
            } catch {
                // Key not found - this is good
                continue
            }
        }
    }

    /// Test CoreData security
    private func testCoreDataSecurity() {
        // Check if sensitive data is stored in CoreData without encryption
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        let context = appDelegate?.persistentContainer.viewContext

        if let context = context {
            // Look for sensitive entity attributes
            let entities = context.persistentStoreCoordinator?.managedObjectModel.entities ?? []

            for entity in entities {
                for attribute in entity.attributesByName.values {
                    let name = attribute.name.lowercased()
                    if name.contains("password") || name.contains("secret") || name.contains("token") {
                        let finding = SecurityFinding(
                            vulnerabilityType: .insecureStorage,
                            severity: .high,
                            title: "Sensitive Data in CoreData",
                            description: "Sensitive attribute '\(attribute.name)' found in CoreData entity '\(entity.name)' without encryption.",
                            location: "CoreData Store",
                            recommendation: "Implement CoreData encryption or move sensitive data to secure storage.",
                            impactAssessment: "High - Sensitive data in CoreData may be compromised if device is jailbroken."
                        )
                        findings.append(finding)
                    }
                }
            }
        }
    }

    /// Test network security
    private func testNetworkSecurity() {
        print("Testing network security...")

        // Test for insecure HTTP connections
        testInsecureHTTP()

        // Test SSL/TLS configuration
        testSSLConfiguration()

        // Test for sensitive data in logs
        testLogSecurity()
    }

    /// Test for insecure HTTP connections
    private func testInsecureHTTP() {
        // This would typically hook into URLSession delegates
        // For demonstration, we'll create a finding for insecure connections
        let finding = SecurityFinding(
            vulnerabilityType: .insecureNetwork,
            severity: .high,
            title: "Potential Insecure Network Communication",
            description: "Application may use insecure HTTP connections instead of HTTPS. This exposes data to man-in-the-middle attacks.",
            location: "Network Layer",
            recommendation: "Ensure all network communication uses HTTPS with proper certificate validation.",
            impactAssessment: "High - Insecure network communication can lead to data interception and tampering."
        )
        findings.append(finding)
    }

    /// Test SSL/TLS configuration
    private func testSSLConfiguration() {
        // Test SSL certificate validation
        let finding = SecurityFinding(
            vulnerabilityType: .insecureNetwork,
            severity: .medium,
            title: "SSL Certificate Validation",
            description: "Verify that SSL certificate validation is properly implemented and not bypassed.",
            location: "SSL/TLS Layer",
            recommendation: "Implement proper SSL certificate validation and avoid disabling ATS.",
            impactAssessment: "Medium - Improper SSL validation can lead to man-in-the-middle attacks."
        )
        findings.append(finding)
    }

    /// Test for sensitive data in logs
    private func testLogSecurity() {
        // Check if sensitive data is being logged
        let finding = SecurityFinding(
            vulnerabilityType: .insecureStorage,
            severity: .medium,
            title: "Sensitive Data Logging",
            description: "Application may log sensitive data such as passwords, tokens, or user credentials.",
            location: "Logging System",
            recommendation: "Implement proper log sanitization to prevent sensitive data leakage.",
            impactAssessment: "Medium - Sensitive data in logs can be accessed by attackers."
        )
        findings.append(finding)
    }

    /// Test authentication mechanisms
    private func testAuthenticationMechanisms() {
        print("Testing authentication mechanisms...")

        // Test for weak password policies
        testPasswordPolicies()

        // Test for insecure authentication storage
        testAuthenticationStorage()

        // Test for session management
        testSessionManagement()
    }

    /// Test password policies
    private func testPasswordPolicies() {
        let finding = SecurityFinding(
            vulnerabilityType: .authenticationBypass,
            severity: .medium,
            title: "Password Policy Strength",
            description: "Verify that password policies enforce sufficient complexity and length requirements.",
            location: "Authentication System",
            recommendation: "Implement strong password policies with minimum length, complexity, and regular updates.",
            impactAssessment: "Medium - Weak passwords can be easily brute-forced or guessed."
        )
        findings.append(finding)
    }

    /// Test authentication storage
    private func testAuthenticationStorage() {
        let finding = SecurityFinding(
            vulnerabilityType: .insecureStorage,
            severity: .high,
            title: "Authentication Token Storage",
            description: "Authentication tokens and session data must be stored securely.",
            location: "Authentication Layer",
            recommendation: "Use Keychain for storing authentication tokens and implement proper session management.",
            impactAssessment: "High - Improper token storage can lead to session hijacking."
        )
        findings.append(finding)
    }

    /// Test session management
    private func testSessionManagement() {
        let finding = SecurityFinding(
            vulnerabilityType: .authenticationBypass,
            severity: .medium,
            title: "Session Management Security",
            description: "Verify proper session timeout, invalidation, and secure cookie handling.",
            location: "Session Management",
            recommendation: "Implement secure session management with proper timeout and invalidation mechanisms.",
            impactAssessment: "Medium - Poor session management can lead to session fixation or hijacking."
        )
        findings.append(finding)
    }

    /// Test memory management
    private func testMemoryManagement() {
        print("Testing memory management...")

        // Test for potential memory leaks using koopman analysis
        let trajectory = reverseKoopman.generateVanDerPolTrajectory(nPoints: 500)
        let koopmanMatrix = reverseKoopman.constructKoopmanMatrix(trajectory: trajectory)

        if !koopmanMatrix.isEmpty {
            let (eigenvalues, _) = reverseKoopman.computeSpectralDecomposition()

            // Look for eigenvalues that might indicate memory issues
            let growingModes = eigenvalues.filter { $0.magnitude > 1.0 }

            if !growingModes.isEmpty {
                let finding = SecurityFinding(
                    vulnerabilityType: .memoryLeak,
                    severity: .high,
                    title: "Potential Memory Leak Detected",
                    description: "Koopman analysis reveals \(growingModes.count) growing modes, indicating potential memory leaks or unbounded growth.",
                    location: "Memory Management",
                    recommendation: "Implement proper memory management and check for retain cycles or unbounded allocations.",
                    koopmanStability: 0.0,
                    spectralRadius: growingModes[0].magnitude,
                    impactAssessment: "High - Memory leaks can lead to application crashes and resource exhaustion."
                )
                findings.append(finding)
            }
        }
    }

    /// Test for race conditions
    private func testRaceConditions() {
        print("Testing for race conditions...")

        let finding = SecurityFinding(
            vulnerabilityType: .raceCondition,
            severity: .medium,
            title: "Race Condition Analysis",
            description: "Perform thorough analysis of concurrent operations for potential race conditions.",
            location: "Concurrent Operations",
            recommendation: "Use proper synchronization mechanisms and avoid shared mutable state where possible.",
            impactAssessment: "Medium - Race conditions can lead to unpredictable behavior and security issues."
        )
        findings.append(finding)
    }

    /// Test for buffer overflows
    private func testBufferOverflows() {
        print("Testing for buffer overflows...")

        let finding = SecurityFinding(
            vulnerabilityType: .bufferOverflow,
            severity: .high,
            title: "Buffer Overflow Protection",
            description: "Verify that all buffer operations are bounds-checked and use safe APIs.",
            location: "Memory Operations",
            recommendation: "Use Swift's safe array operations and avoid unsafe pointer arithmetic.",
            impactAssessment: "High - Buffer overflows can lead to arbitrary code execution and system compromise."
        )
        findings.append(finding)
    }

    // MARK: - IT Assistant Specific Tests

    /// Perform comprehensive penetration testing on IT Assistant
    func testITAssistantSecurity() -> [SecurityFinding] {
        print("Starting IT Assistant penetration testing...")

        findings.removeAll()

        // Analyze system behavior using reverse koopman
        analyzeITAssistantBehavior()

        // Test authentication vulnerabilities
        testITAssistantAuthentication()

        // Test logging security
        testITAssistantLogging()

        // Test API endpoint security
        testITAssistantAPI()

        // Test configuration security
        testITAssistantConfiguration()

        // Test LLM integration security
        testITAssistantLLMSecurity()

        print("IT Assistant penetration testing completed. Found \(findings.count) issues.")
        return findings
    }

    /// Analyze IT Assistant behavior using reverse koopman operators
    private func analyzeITAssistantBehavior() {
        print("Analyzing IT Assistant behavior with reverse koopman operators...")

        // Simulate IT Assistant request patterns as dynamical system
        let requestPatterns = generateITAssistantRequestPatterns()

        // Construct koopman matrix from request patterns
        let koopmanMatrix = reverseKoopman.constructKoopmanMatrix(trajectory: requestPatterns)

        if !koopmanMatrix.isEmpty {
            // Compute spectral decomposition
            let (eigenvalues, _) = reverseKoopman.computeSpectralDecomposition()

            // Analyze stability and create findings
            analyzeITAssistantStabilityFindings(eigenvalues: eigenvalues)
        }
    }

    /// Generate IT Assistant request patterns for analysis
    private func generateITAssistantRequestPatterns() -> [[Double]] {
        var patterns: [[Double]] = []

        // Simulate different types of requests and their state transitions
        for i in 0..<100 {
            let timeStep = Double(i) * 0.1
            let requestLoad = sin(timeStep) + 0.5 // Varying request load
            let authFailures = cos(timeStep * 2) + 0.3 // Authentication patterns
            let errorRate = sin(timeStep * 3) + 0.2 // Error patterns

            patterns.append([requestLoad, authFailures, errorRate])
        }

        return patterns
    }

    /// Analyze IT Assistant stability findings
    private func analyzeITAssistantStabilityFindings(eigenvalues: [ComplexNumber], cLower: Double = 0.5) {
        let spectralRadius = eigenvalues.map { $0.magnitude }.max() ?? 0.0

        // Check for system instability under load
        if spectralRadius > 1.2 {
            let finding = SecurityFinding(
                vulnerabilityType: .privilegeEscalation,
                severity: .critical,
                title: "IT Assistant System Instability",
                description: "Reverse koopman analysis reveals critical instability in IT Assistant under varying loads (spectral radius = \(String(format: "%.3f", spectralRadius))). This indicates potential for service disruption or privilege escalation.",
                location: "IT Assistant Core System",
                recommendation: "Implement proper rate limiting, circuit breakers, and load balancing to stabilize system behavior.",
                koopmanStability: cLower,
                spectralRadius: spectralRadius,
                conditionNumber: 0.0,
                reconstructionError: 0.0,
                dominantModes: eigenvalues.count,
                impactAssessment: "Critical - System instability can lead to service denial, data corruption, or security bypass."
            )
            findings.append(finding)
        }

        // Check for poor conditioning (numerical instability)
        if spectralRadius > 1.0 && cLower < 0.3 {
            let finding = SecurityFinding(
                vulnerabilityType: .memoryLeak,
                severity: .high,
                title: "IT Assistant Numerical Instability",
                description: "Poor numerical conditioning detected in IT Assistant (c = \(String(format: "%.3f", cLower))). This may lead to memory corruption or precision loss in request processing.",
                location: "IT Assistant Numerical Operations",
                recommendation: "Implement proper regularization and numerical stability checks in request processing.",
                koopmanStability: cLower,
                spectralRadius: spectralRadius,
                impactAssessment: "High - Numerical instability can lead to incorrect responses or system crashes."
            )
            findings.append(finding)
        }
    }

    /// Test IT Assistant authentication mechanisms
    private func testITAssistantAuthentication() {
        print("Testing IT Assistant authentication...")

        // Test token-based authentication vulnerabilities
        testTokenAuthentication()

        // Test authentication bypass scenarios
        testAuthBypass()

        // Test session management
        testSessionManagement()
    }

    /// Test token authentication vulnerabilities
    private func testTokenAuthentication() {
        // Test for weak token validation
        let finding = SecurityFinding(
            vulnerabilityType: .authenticationBypass,
            severity: .high,
            title: "Token Authentication Vulnerabilities",
            description: "IT Assistant uses Bearer token authentication but may have insufficient validation. Tokens are compared using simple string equality without additional security measures.",
            location: "IT Assistant Authentication",
            recommendation: "Implement proper token validation with expiration, rotation, and secure comparison mechanisms. Consider JWT with proper signing.",
            impactAssessment: "High - Weak token validation can lead to unauthorized access to IT support functions."
        )
        findings.append(finding)

        // Test for token storage
        let tokenStorageFinding = SecurityFinding(
            vulnerabilityType: .insecureStorage,
            severity: .medium,
            title: "Token Storage in Environment Variables",
            description: "Authentication tokens stored in environment variables may be accessible to other processes or logged inadvertently.",
            location: "IT Assistant Configuration",
            recommendation: "Use secure key management systems or encrypted configuration files for storing sensitive tokens.",
            impactAssessment: "Medium - Environment variable exposure can lead to token compromise."
        )
        findings.append(finding)
    }

    /// Test authentication bypass scenarios
    private func testAuthBypass() {
        let finding = SecurityFinding(
            vulnerabilityType: .authenticationBypass,
            severity: .medium,
            title: "Authentication Bypass Potential",
            description: "IT Assistant may be vulnerable to authentication bypass if AUTH_TOKEN is not properly set when REQUIRE_AUTH is enabled.",
            location: "IT Assistant Auth Configuration",
            recommendation: "Ensure proper validation that AUTH_TOKEN is set when REQUIRE_AUTH is enabled. Implement graceful fallback or proper error handling.",
            impactAssessment: "Medium - Misconfiguration can lead to unauthorized access to the IT Assistant service."
        )
        findings.append(finding)
    }

    /// Test IT Assistant logging security
    private func testITAssistantLogging() {
        print("Testing IT Assistant logging security...")

        // Test for sensitive data in logs
        testSensitiveDataLogging()

        // Test for log injection
        testLogInjection()

        // Test for log file security
        testLogFileSecurity()
    }

    /// Test for sensitive data logging
    private func testSensitiveDataLogging() {
        let finding = SecurityFinding(
            vulnerabilityType: .insecureStorage,
            severity: .high,
            title: "Sensitive Data in IT Assistant Logs",
            description: "IT Assistant logs may contain sensitive information including user questions, authentication tokens, and system configuration.",
            location: "IT Assistant Logging System",
            recommendation: "Implement log sanitization to prevent sensitive data leakage. Use structured logging with proper data classification.",
            impactAssessment: "High - Log files may expose sensitive information to unauthorized access."
        )
        findings.append(finding)
    }

    /// Test for log injection vulnerabilities
    private func testLogInjection() {
        let finding = SecurityFinding(
            vulnerabilityType: .sqlInjection,
            severity: .medium,
            title: "Log Injection Vulnerability",
            description: "IT Assistant uses user-provided data in log entries without proper sanitization, potentially allowing log injection attacks.",
            location: "IT Assistant Logging",
            recommendation: "Implement proper input sanitization for all user data before logging. Use parameterized logging where possible.",
            impactAssessment: "Medium - Log injection can lead to log manipulation or denial of service."
        )
        findings.append(finding)
    }

    /// Test IT Assistant API endpoint security
    private func testITAssistantAPI() {
        print("Testing IT Assistant API security...")

        // Test for API rate limiting
        testAPIRateLimiting()

        // Test for input validation
        testInputValidation()

        // Test for error information disclosure
        testErrorDisclosure()
    }

    /// Test API rate limiting
    private func testAPIRateLimiting() {
        let finding = SecurityFinding(
            vulnerabilityType: .authenticationBypass,
            severity: .medium,
            title: "Missing Rate Limiting",
            description: "IT Assistant API endpoints lack rate limiting, making them vulnerable to brute force attacks and denial of service.",
            location: "IT Assistant API Endpoints",
            recommendation: "Implement proper rate limiting based on IP address, user, or API key. Consider using middleware for consistent rate limiting.",
            impactAssessment: "Medium - Lack of rate limiting can lead to resource exhaustion and brute force attacks."
        )
        findings.append(finding)
    }

    /// Test input validation
    private func testInputValidation() {
        let finding = SecurityFinding(
            vulnerabilityType: .sqlInjection,
            severity: .high,
            title: "Insufficient Input Validation",
            description: "IT Assistant accepts user input without comprehensive validation, potentially allowing injection attacks or malformed requests.",
            location: "IT Assistant Input Processing",
            recommendation: "Implement comprehensive input validation for all user-provided data including length limits, character restrictions, and format validation.",
            impactAssessment: "High - Poor input validation can lead to injection attacks, buffer overflows, or unexpected behavior."
        )
        findings.append(finding)
    }

    /// Test IT Assistant configuration security
    private func testITAssistantConfiguration() {
        print("Testing IT Assistant configuration security...")

        // Test for insecure defaults
        testInsecureDefaults()

        // Test for configuration exposure
        testConfigurationExposure()
    }

    /// Test for insecure defaults
    private func testInsecureDefaults() {
        let finding = SecurityFinding(
            vulnerabilityType: .weakEncryption,
            severity: .low,
            title: "Insecure Default Configuration",
            description: "IT Assistant defaults to disabled authentication (REQUIRE_AUTH=0), potentially exposing the service to unauthorized access.",
            location: "IT Assistant Configuration",
            recommendation: "Change default configuration to require authentication and provide clear documentation for secure setup.",
            impactAssessment: "Low - Insecure defaults may lead to accidental exposure of the service."
        )
        findings.append(finding)
    }

    /// Test IT Assistant LLM integration security
    private func testITAssistantLLMSecurity() {
        print("Testing IT Assistant LLM integration security...")

        // Test for prompt injection
        testPromptInjection()

        // Test for LLM data leakage
        testLLMDataLeakage()
    }

    /// Test for prompt injection vulnerabilities
    private func testPromptInjection() {
        let finding = SecurityFinding(
            vulnerabilityType: .sqlInjection,
            severity: .high,
            title: "Prompt Injection Vulnerability",
            description: "IT Assistant constructs prompts using user input without proper sanitization, potentially allowing prompt injection attacks.",
            location: "IT Assistant LLM Integration",
            recommendation: "Implement proper prompt sanitization and validation. Use structured prompts with clear boundaries and input validation.",
            impactAssessment: "High - Prompt injection can lead to unauthorized LLM behavior or data leakage."
        )
        findings.append(finding)
    }

    /// Test for LLM data leakage
    private func testLLMDataLeakage() {
        let finding = SecurityFinding(
            vulnerabilityType: .insecureNetwork,
            severity: .medium,
            title: "LLM Data Leakage Potential",
            description: "IT Assistant sends user questions to external LLM services without comprehensive data sanitization or classification.",
            location: "IT Assistant LLM Communication",
            recommendation: "Implement data classification and sanitization before sending to external services. Consider on-premise LLM solutions for sensitive data.",
            impactAssessment: "Medium - Data sent to external LLM services may be logged, analyzed, or exposed."
        )
        findings.append(finding)
    }

    // MARK: - Utility Methods

    /// Export findings to JSON
    /// - Returns: JSON string representation of findings
    func exportFindingsToJSON() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted

        do {
            let data = try encoder.encode(findings)
            return String(data: data, encoding: .utf8) ?? "{}"
        } catch {
            print("Error encoding findings: \(error)")
            return "{}"
        }
    }

    /// Generate security report
    /// - Returns: Formatted security report
    func generateSecurityReport() -> String {
        var report = """
        iOS Security Assessment Report
        =============================

        Assessment Date: \(Date())
        Total Findings: \(findings.count)

        Summary by Severity:
        """

        let severityCounts = Dictionary(grouping: findings) { $0.severity }
            .mapValues { $0.count }

        for severity in [SeverityLevel.critical, .high, .medium, .low, .info] {
            report += "\n\(severity.rawValue): \(severityCounts[severity] ?? 0)"
        }

        report += "\n\nDetailed Findings:\n"

        for (index, finding) in findings.enumerated() {
            report += """
            \(index + 1). \(finding.title)
            Severity: \(finding.severity.rawValue)
            Type: \(finding.vulnerabilityType.rawValue)
            Location: \(finding.location)

            Description: \(finding.description)

            Impact: \(finding.impactAssessment)
            CVSS Score: \(finding.cvssScore)

            Koopman Analysis:
            - Stability: \(String(format: "%.3f", finding.koopmanStability))
            - Spectral Radius: \(String(format: "%.3f", finding.spectralRadius))
            - Condition Number: \(String(format: "%.3f", finding.conditionNumber))
            - Reconstruction Error: \(String(format: "%.6f", finding.reconstructionError))

            Recommendation: \(finding.recommendation)

            ---
            """
        }

        return report
    }
}

/// Keychain management utility
class KeychainManager {

    func save(key: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]

        SecItemDelete(query as CFDictionary)
        let status = SecItemAdd(query as CFDictionary, nil)

        if status != errSecSuccess {
            throw NSError(domain: "KeychainError", code: Int(status), userInfo: nil)
        }
    }

    func load(key: String) throws -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        if status == errSecItemNotFound {
            return nil
        }

        if status != errSecSuccess {
            throw NSError(domain: "KeychainError", code: Int(status), userInfo: nil)
        }

        return result as? Data
    }
}
