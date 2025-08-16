import Foundation
import Combine

#if canImport(HealthKit)
import HealthKit
#endif

final class CognitiveTracker: ObservableObject {
    @Published var hrvSDNNms: Double?

    #if canImport(HealthKit)
    private let health = HKHealthStore()
    #endif

    func requestAuthorization(completion: @escaping (AppError?) -> Void) {
        #if canImport(HealthKit)
        guard HKHealthStore.isHealthDataAvailable() else { completion(.healthDataUnavailable); return }
        guard let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { completion(.healthDataUnavailable); return }
        health.requestAuthorization(toShare: [], read: [hrvType]) { success, _ in
            DispatchQueue.main.async { completion(success ? nil : .healthDataUnavailable) }
        }
        #else
        completion(nil)
        #endif
    }

    func fetchLatest() {
        #if canImport(HealthKit)
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return }
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrvType, predicate: nil, limit: 1, sortDescriptors: [sort]) { [weak self] _, samples, _ in
            guard let sample = samples?.first as? HKQuantitySample else { return }
            let value = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            DispatchQueue.main.async { self?.hrvSDNNms = value }
        }
        health.execute(query)
        #else
        // Simulate HRV on platforms without HealthKit
        hrvSDNNms = Double.random(in: 25...80)
        #endif
    }
}