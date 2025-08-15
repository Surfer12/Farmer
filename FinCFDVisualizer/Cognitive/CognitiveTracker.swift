import Foundation

#if os(iOS)
import HealthKit
#endif

final class CognitiveTracker {
    static let shared = CognitiveTracker()
    private init() {}

    #if os(iOS)
    private let healthStore = HKHealthStore()
    #endif

    func fetchHRV(completion: @escaping (Double?) -> Void) {
        #if os(iOS)
        guard HKHealthStore.isHealthDataAvailable(),
              let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(nil)
            return
        }
        healthStore.requestAuthorization(toShare: [], read: [hrvType]) { [weak self] granted, _ in
            guard granted, let self else { completion(nil); return }
            let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            let query = HKSampleQuery(sampleType: hrvType, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else { completion(nil); return }
                let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                completion(ms)
            }
            self.healthStore.execute(query)
        }
        #else
        completion(nil)
        #endif
    }
}