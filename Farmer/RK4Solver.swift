import Foundation

// MARK: - RK4 Solver for Burgers' Equation
class RK4Solver {
    private let nu: Double // Viscosity parameter
    private let dx: Double // Spatial step size
    private let dt: Double // Time step size
    
    init(viscosity: Double = 0.01, spatialStep: Double = 0.05, timeStep: Double = 0.001) {
        self.nu = viscosity
        self.dx = spatialStep
        self.dt = timeStep
    }
    
    // Initial condition: u(x, 0) = -sin(π*x)
    func initialCondition(_ x: Double) -> Double {
        return -sin(.pi * x)
    }
    
    // Spatial derivative approximation (central difference)
    private func spatialDerivative(_ u: [Double], _ i: Int) -> Double {
        let n = u.count
        if i == 0 {
            return (u[1] - u[n-1]) / (2 * dx) // Periodic boundary
        } else if i == n - 1 {
            return (u[0] - u[n-2]) / (2 * dx) // Periodic boundary
        } else {
            return (u[i+1] - u[i-1]) / (2 * dx)
        }
    }
    
    // Second spatial derivative (central difference)
    private func secondSpatialDerivative(_ u: [Double], _ i: Int) -> Double {
        let n = u.count
        if i == 0 {
            return (u[1] - 2*u[0] + u[n-1]) / (dx * dx) // Periodic boundary
        } else if i == n - 1 {
            return (u[0] - 2*u[n-1] + u[n-2]) / (dx * dx) // Periodic boundary
        } else {
            return (u[i+1] - 2*u[i] + u[i-1]) / (dx * dx)
        }
    }
    
    // Burgers' equation RHS: -u * u_x + ν * u_xx
    private func burgersRHS(_ u: [Double]) -> [Double] {
        var dudt = [Double](repeating: 0.0, count: u.count)
        
        for i in 0..<u.count {
            let u_x = spatialDerivative(u, i)
            let u_xx = secondSpatialDerivative(u, i)
            dudt[i] = -u[i] * u_x + nu * u_xx
        }
        
        return dudt
    }
    
    // RK4 step
    private func rk4Step(_ u: [Double]) -> [Double] {
        let k1 = burgersRHS(u)
        
        let u2 = zip(u, k1).map { $0 + dt * $1 / 2.0 }
        let k2 = burgersRHS(u2)
        
        let u3 = zip(u, k2).map { $0 + dt * $1 / 2.0 }
        let k3 = burgersRHS(u3)
        
        let u4 = zip(u, k3).map { $0 + dt * $1 }
        let k4 = burgersRHS(u4)
        
        return zip4(u, k1, k2, k3, k4).map { u, k1, k2, k3, k4 in
            u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        }
    }
    
    // Solve Burgers' equation from t=0 to t=finalTime
    func solve(xPoints: [Double], finalTime: Double) -> [Double] {
        let nSteps = Int(finalTime / dt)
        
        // Initialize solution with initial condition
        var u = xPoints.map { initialCondition($0) }
        
        // Time stepping
        for step in 0..<nSteps {
            u = rk4Step(u)
            
            // Apply boundary conditions (u(-1,t) = u(1,t) = 0)
            if let firstIdx = xPoints.firstIndex(of: -1.0) {
                u[firstIdx] = 0.0
            }
            if let lastIdx = xPoints.firstIndex(of: 1.0) {
                u[lastIdx] = 0.0
            }
        }
        
        return u
    }
    
    // Solve and return full time evolution
    func solveWithHistory(xPoints: [Double], finalTime: Double, saveInterval: Int = 100) -> [(time: Double, solution: [Double])] {
        let nSteps = Int(finalTime / dt)
        var u = xPoints.map { initialCondition($0) }
        var history: [(time: Double, solution: [Double])] = []
        
        // Save initial condition
        history.append((time: 0.0, solution: u))
        
        for step in 0..<nSteps {
            u = rk4Step(u)
            
            // Apply boundary conditions
            if let firstIdx = xPoints.firstIndex(of: -1.0) {
                u[firstIdx] = 0.0
            }
            if let lastIdx = xPoints.firstIndex(of: 1.0) {
                u[lastIdx] = 0.0
            }
            
            // Save solution at intervals
            if step % saveInterval == 0 {
                let currentTime = Double(step) * dt
                history.append((time: currentTime, solution: u))
            }
        }
        
        // Save final solution
        history.append((time: finalTime, solution: u))
        
        return history
    }
}

// MARK: - Utility Functions
private func zip4<T>(_ a: [T], _ b: [T], _ c: [T], _ d: [T], _ e: [T]) -> [(T, T, T, T, T)] {
    return Array(zip(zip(zip(zip(a, b), c), d), e).map { ((($0.0.0, $0.0.1), $0.1), $1) }.map { (($0.0.0, $0.0.1, $0.1), $1) }.map { ($0.0, $0.1, $1.0, $1.1, $1.2) })
}

// MARK: - Comparison Analysis
struct PINNRKComparison {
    static func compareAtTime(
        pinnModel: PINN,
        rk4Solver: RK4Solver,
        xPoints: [Double],
        time: Double
    ) -> (pinnSolution: [Double], rk4Solution: [Double], mse: Double, maxError: Double) {
        
        // Get PINN solution
        let pinnSolution = xPoints.map { x in pinnModel.forward(x, time) }
        
        // Get RK4 solution
        let rk4Solution = rk4Solver.solve(xPoints: xPoints, finalTime: time)
        
        // Compute errors
        let errors = zip(pinnSolution, rk4Solution).map { abs($0 - $1) }
        let mse = errors.map { $0 * $0 }.reduce(0, +) / Double(errors.count)
        let maxError = errors.max() ?? 0.0
        
        return (pinnSolution, rk4Solution, mse, maxError)
    }
    
    // Generate sample data for visualization
    static func generateVisualizationData(
        pinnModel: PINN,
        rk4Solver: RK4Solver,
        time: Double = 1.0
    ) -> (xPoints: [Double], pinnData: [Double], rk4Data: [Double]) {
        
        let xPoints = Array(stride(from: -1.0, through: 1.0, by: 0.1))
        let comparison = compareAtTime(
            pinnModel: pinnModel,
            rk4Solver: rk4Solver,
            xPoints: xPoints,
            time: time
        )
        
        return (xPoints, comparison.pinnSolution, comparison.rk4Solution)
    }
    
    // Analyze BNSL implications
    static func analyzeBNSL(
        trainingHistory: [(epoch: Int, loss: Double, psi: Double)]
    ) -> (inflectionPoints: [Int], scalingBehavior: String) {
        
        var inflectionPoints: [Int] = []
        let losses = trainingHistory.map { $0.loss }
        
        // Find inflection points in loss curve
        for i in 1..<losses.count-1 {
            let prev = losses[i-1]
            let curr = losses[i]
            let next = losses[i+1]
            
            // Check for change in curvature (second derivative sign change)
            let d1_prev = curr - prev
            let d1_next = next - curr
            let d2 = d1_next - d1_prev
            
            if i > 1 {
                let prev_d2 = d1_prev - (prev - losses[i-2])
                if (d2 > 0 && prev_d2 < 0) || (d2 < 0 && prev_d2 > 0) {
                    inflectionPoints.append(trainingHistory[i].epoch)
                }
            }
        }
        
        // Analyze scaling behavior
        let finalLoss = losses.last ?? 0.0
        let initialLoss = losses.first ?? 1.0
        let reductionRatio = finalLoss / initialLoss
        
        var scalingBehavior: String
        if reductionRatio < 0.01 {
            scalingBehavior = "Exponential decay - strong BNSL power law"
        } else if reductionRatio < 0.1 {
            scalingBehavior = "Power law decay - moderate BNSL behavior"
        } else if inflectionPoints.count > 2 {
            scalingBehavior = "Non-monotonic - characteristic BNSL inflection points"
        } else {
            scalingBehavior = "Linear decay - weak BNSL signature"
        }
        
        return (inflectionPoints, scalingBehavior)
    }
}