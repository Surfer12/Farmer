// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  ContentView.swift
//  Farmer
//
//  Created by Ryan David Oates on 8/7/25.
//

import SwiftUI
import CoreData

struct ContentView: View {
    @Environment(\.managedObjectContext) private var viewContext

    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Item.timestamp, ascending: true)],
        animation: .default)
    private var items: FetchedResults<Item>

    var body: some View {
        NavigationView {
            List {
                Section("PINN") {
                    NavigationLink {
                        PINNDemoView()
                    } label: {
                        Text("PINN: Burgers' (PINN vs RK4)")
                    }
                }
                ForEach(items) { item in
                    NavigationLink {
                        Text("Item at \(item.timestamp!, formatter: itemFormatter)")
                    } label: {
                        Text(item.timestamp!, formatter: itemFormatter)
                    }
                }
                .onDelete(perform: deleteItems)
            }
            .toolbar {
#if os(iOS)
                ToolbarItem(placement: .navigationBarTrailing) {
                    EditButton()
                }
#endif
                ToolbarItem {
                    Button(action: addItem) {
                        Label("Add Item", systemImage: "plus")
                    }
                }
            }
            Text("Select an item")
        }
    }

    private func addItem() {
        withAnimation {
            let newItem = Item(context: viewContext)
            newItem.timestamp = Date()

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }

    private func deleteItems(offsets: IndexSet) {
        withAnimation {
            offsets.map { items[$0] }.forEach(viewContext.delete)

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }
}

private let itemFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.dateStyle = .short
    formatter.timeStyle = .medium
    return formatter
}()

#Preview {
    ContentView().environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
}

// MARK: - PINN Demo (Training + Visualization)

import Foundation
#if canImport(Charts)
import Charts
#endif

struct PINNDemoView: View {
    @StateObject private var engine = PINNDemoEngine()

    var body: some View {
        VStack(spacing: 16) {
#if canImport(Charts)
            Chart {
                ForEach(engine.chartPointsPINN) { p in
                    LineMark(
                        x: .value("x", p.x),
                        y: .value("u", p.y)
                    )
                    .foregroundStyle(.blue)
                    .interpolationMethod(.catmullRom)
                }
                ForEach(engine.chartPointsRK4) { p in
                    LineMark(
                        x: .value("x", p.x),
                        y: .value("u", p.y)
                    )
                    .foregroundStyle(.red)
                    .interpolationMethod(.catmullRom)
                }
            }
            .frame(height: 260)
#else
            // Fallback: simple values list when Charts is unavailable
            ScrollView {
                VStack(alignment: .leading) {
                    Text("Charts not available; displaying sample values.")
                    Text("x, PINN, RK4")
                    ForEach(Array(engine.x.enumerated()), id: \.offset) { idx, xv in
                        Text(String(format: "%.2f\t%.3f\t%.3f", xv, engine.pinnU[idx], engine.rk4U[idx]))
                    }
                }
            }
            .frame(height: 260)
#endif

            HStack {
                Text(String(format: "Epoch: %d", engine.epoch))
                Spacer()
                Text(String(format: "Loss: %.4f", engine.loss))
            }

            // Hybrid output and UQ metrics
            VStack(spacing: 6) {
                Text(String(format: "Ψ(x): %.3f", engine.psi))
                Text(String(format: "S(x): %.2f   N(x): %.2f   α(t): %.2f", engine.Sx, engine.Nx, engine.alpha))
                Text(String(format: "R_cognitive: %.3f   R_efficiency: %.3f", engine.R_cognitive, engine.R_efficiency))
            }

            HStack {
                Button("Train 1000 epochs") {
                    engine.train(epochs: 1000, printEvery: 50)
                }
                .buttonStyle(.borderedProminent)

                Button("Reset") {
                    engine.reset()
                }
                .buttonStyle(.bordered)
            }

            Spacer(minLength: 0)
        }
        .padding()
        .navigationTitle("PINN vs RK4")
    }
}

private struct LinePoint: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
}

@MainActor
final class PINNDemoEngine: ObservableObject {
    // Grid
    @Published var x: [Double]
    // Solutions
    @Published var rk4U: [Double]
    @Published var pinnU: [Double]
    // Training state
    @Published var epoch: Int = 0
    @Published var loss: Double = 0
    // Hybrid metrics
    @Published var Sx: Double = 0
    @Published var Nx: Double = 0
    @Published var alpha: Double = 0.5
    @Published var psi: Double = 0
    @Published var R_cognitive: Double = 0
    @Published var R_efficiency: Double = 0.10

    private var prevLoss: Double = .infinity
    private let learningRate: Double = 0.005
    private let lambdaSmooth: Double = 0.01
    private var mlp = SimpleMLP(inputSize: 1, hiddenSize: 16, outputSize: 1, seed: 42)

    var chartPointsPINN: [LinePoint] { zip(x, pinnU).map { LinePoint(x: $0.0, y: $0.1) } }
    var chartPointsRK4: [LinePoint] { zip(x, rk4U).map { LinePoint(x: $0.0, y: $0.1) } }

    init() {
        self.x = Array(stride(from: -1.0, through: 1.0, by: 0.1))
        let dx = x[1] - x[0]
        let u0 = x.map { -sin(.pi * $0) }
        self.rk4U = BurgersSolverRK4.solve(u0: u0, dx: dx, nu: 0.01, tFinal: 1.0, dt: 0.001)
        self.pinnU = x.map { mlp.predict(x: [$0])[0] }
        computeMetrics(currentLoss: mse(pred: pinnU, target: rk4U))
    }

    func train(epochs: Int, printEvery: Int) {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            let xs = self.x
            let ys = self.rk4U
            let dx = xs[1] - xs[0]
            var currentLoss = self.mse(pred: self.pinnU, target: ys)
            for e in 1...epochs {
                let result = self.mlp.trainBatch(xs: xs, targets: ys, smoothingDx: dx, lambdaSmooth: self.lambdaSmooth, learningRate: self.learningRate)
                let yhat = xs.map { self.mlp.predict(x: [$0])[0] }
                currentLoss = result.totalLoss

                await MainActor.run {
                    self.pinnU = yhat
                    self.epoch += 1
                    self.loss = currentLoss
                    self.computeMetrics(currentLoss: currentLoss, mseLoss: result.mse, smoothLoss: result.smooth)
                }

                if e % printEvery == 0 {
                    print(String(format: "Epoch %4d  total=%.6f  mse=%.6f  smooth=%.6f", e, currentLoss, result.mse, result.smooth))
                }

                try? await Task.sleep(nanoseconds: 500_000) // yield
            }
        }
    }

    func reset() {
        mlp = SimpleMLP(inputSize: 1, hiddenSize: 16, outputSize: 1, seed: 43)
        pinnU = x.map { mlp.predict(x: [$0])[0] }
        epoch = 0
        loss = mse(pred: pinnU, target: rk4U)
        prevLoss = .infinity
        Sx = 0; Nx = 0; alpha = 0.5; psi = 0; R_cognitive = 0; R_efficiency = 0.10
    }

    private func computeMetrics(currentLoss: Double, mseLoss: Double? = nil, smoothLoss: Double? = nil) {
        let mseVal = mseLoss ?? currentLoss
        let smoothVal = smoothLoss ?? 0.0
        // S(x): higher is better; map from MSE to [0,1]
        Sx = 1.0 / (1.0 + max(mseVal, 0))
        // R_cognitive from smoothness (proxy for PDE residual)
        R_cognitive = min(1.0, smoothVal)
        // N(x): convergence rate from relative loss drop
        if prevLoss.isFinite {
            let rel = (prevLoss - currentLoss) / max(prevLoss, 1e-8)
            Nx = max(0.0, min(1.0, rel))
        } else {
            Nx = 0
        }
        prevLoss = currentLoss
        // α(t): fixed for now; could be adapted by validation feedback
        alpha = 0.5
        let oHybrid = (1.0 - alpha) * Sx + alpha * Nx
        let lambda1 = 0.6, lambda2 = 0.4
        let pTotal = lambda1 * R_cognitive + lambda2 * R_efficiency
        let pBase = 0.80, beta = 1.20
        let pAdj = min(1.0, pBase * beta)
        psi = oHybrid * exp(-pTotal) * pAdj
    }

    private func mse(pred: [Double], target: [Double]) -> Double {
        guard pred.count == target.count, pred.count > 0 else { return 0 }
        let n = Double(pred.count)
        let s = zip(pred, target).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        return s / n
    }
}

// MARK: - Simple 1D MLP with manual backprop (tanh)

private struct SimpleMLP {
    let inputSize: Int
    let hiddenSize: Int
    let outputSize: Int
    var w1: [Double] // hiddenSize
    var b1: [Double] // hiddenSize
    var w2: [Double] // hiddenSize
    var b2: Double

    init(inputSize: Int, hiddenSize: Int, outputSize: Int, seed: UInt64) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        var rng = SplitMix64(seed: seed)
        let lim1 = sqrt(6.0 / Double(inputSize + hiddenSize))
        w1 = (0..<hiddenSize).map { _ in rng.uniform(-lim1, lim1) }
        b1 = Array(repeating: 0.0, count: hiddenSize)
        let lim2 = sqrt(6.0 / Double(hiddenSize + outputSize))
        w2 = (0..<hiddenSize).map { _ in rng.uniform(-lim2, lim2) }
        b2 = 0.0
    }

    func predict(x: [Double]) -> [Double] {
        guard x.count == 1 else { return [0] }
        let xv = x[0]
        let h = (0..<hiddenSize).map { j in tanh(w1[j] * xv + b1[j]) }
        let y = zip(w2, h).reduce(b2, { $0 + $1.0 * $1.1 })
        return [y]
    }

    func forwardBatch(xs: [Double]) -> (h: [[Double]], y: [Double]) {
        let h = xs.map { xv in (0..<hiddenSize).map { j in tanh(w1[j] * xv + b1[j]) } }
        let y = h.map { hj in zip(w2, hj).reduce(b2, { $0 + $1.0 * $1.1 }) }
        return (h, y)
    }

    mutating func trainBatch(xs: [Double], targets: [Double], smoothingDx: Double, lambdaSmooth: Double, learningRate: Double) -> (totalLoss: Double, mse: Double, smooth: Double) {
        let n = xs.count
        let (h, y) = forwardBatch(xs: xs)
        // MSE
        var mseLoss = 0.0
        for i in 0..<n { let d = y[i] - targets[i]; mseLoss += d * d }
        mseLoss /= Double(n)
        // Smoothness (proxy for PDE residual)
        let invDx2 = 1.0 / (smoothingDx * smoothingDx)
        var sVals = Array(repeating: 0.0, count: n)
        for i in 0..<n {
            let ip = (i + 1) % n
            let im = (i - 1 + n) % n
            sVals[i] = (y[ip] - 2.0 * y[i] + y[im]) * invDx2
        }
        var smoothLoss = 0.0
        for i in 0..<n { smoothLoss += sVals[i] * sVals[i] }
        smoothLoss /= Double(n)
        let totalLoss = mseLoss + lambdaSmooth * smoothLoss

        // dL/dy
        var g = Array(repeating: 0.0, count: n)
        for i in 0..<n { g[i] += 2.0 * (y[i] - targets[i]) / Double(n) }
        let c = 2.0 * lambdaSmooth / Double(n)
        for i in 0..<n {
            let ip = (i + 1) % n
            let im = (i - 1 + n) % n
            // derivative of sum s_k^2 w.r.t y[i] equals 2*(s_{i-1} - 2*s_i + s_{i+1})/dx^2 factor folded into sVals
            g[i] += c * (sVals[(i - 1 + n) % n] - 2.0 * sVals[i] + sVals[(i + 1) % n])
        }

        // Accumulate gradients
        var dW2 = Array(repeating: 0.0, count: hiddenSize)
        var dB2 = 0.0
        var dW1 = Array(repeating: 0.0, count: hiddenSize)
        var dB1 = Array(repeating: 0.0, count: hiddenSize)
        for i in 0..<n {
            let xv = xs[i]
            let gi = g[i]
            let hi = h[i]
            // output layer
            for j in 0..<hiddenSize {
                dW2[j] += gi * hi[j]
            }
            dB2 += gi
            // hidden layer
            for j in 0..<hiddenSize {
                let dh = (1.0 - hi[j] * hi[j])
                let common = gi * w2[j] * dh
                dW1[j] += common * xv
                dB1[j] += common
            }
        }
        // Gradient step
        let lr = learningRate
        for j in 0..<hiddenSize {
            w2[j] -= lr * dW2[j]
            w1[j] -= lr * dW1[j]
            b1[j] -= lr * dB1[j]
        }
        b2 -= lr * dB2

        return (totalLoss, mseLoss, smoothLoss)
    }
}

// MARK: - 1D viscous Burgers' solver via method-of-lines + RK4 (periodic BC)

private enum BurgersSolverRK4 {
    static func solve(u0: [Double], dx: Double, nu: Double, tFinal: Double, dt: Double) -> [Double] {
        guard !u0.isEmpty else { return [] }
        var u = u0
        var t = 0.0
        while t < tFinal - 1e-12 {
            let step = min(dt, tFinal - t)
            u = rk4Step(u: u, dt: step, dx: dx, nu: nu)
            t += step
        }
        return u
    }

    private static func rk4Step(u: [Double], dt: Double, dx: Double, nu: Double) -> [Double] {
        let k1 = rhs(u: u, dx: dx, nu: nu)
        let u2 = zip(u, k1).map { $0 + 0.5 * dt * $1 }
        let k2 = rhs(u: u2, dx: dx, nu: nu)
        let u3 = zip(u, k2).map { $0 + 0.5 * dt * $1 }
        let k3 = rhs(u: u3, dx: dx, nu: nu)
        let u4 = zip(u, k3).map { $0 + dt * $1 }
        let k4 = rhs(u: u4, dx: dx, nu: nu)
        var un = Array(repeating: 0.0, count: u.count)
        for i in 0..<u.count { un[i] = u[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) }
        return un
    }

    private static func rhs(u: [Double], dx: Double, nu: Double) -> [Double] {
        let n = u.count
        var dudt = Array(repeating: 0.0, count: n)
        let inv2dx = 1.0 / (2.0 * dx)
        let invDx2 = 1.0 / (dx * dx)
        for i in 0..<n {
            let ip = (i + 1) % n
            let im = (i - 1 + n) % n
            let ux = (u[ip] - u[im]) * inv2dx
            let uxx = (u[im] - 2.0 * u[i] + u[ip]) * invDx2
            // u_t = -u * u_x + nu * u_xx
            dudt[i] = -u[i] * ux + nu * uxx
        }
        return dudt
    }
}

// MARK: - Tiny RNG for reproducible init

private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        var z = state &+ 0x9E3779B97F4A7C15
        state = z
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ a: Double, _ b: Double) -> Double {
        let r = next()
        let u = Double(r) / Double(UInt64.max)
        return a + (b - a) * u
    }
}
