import Foundation
import SwiftUI

enum AppError: LocalizedError, Identifiable, Equatable {
    case healthDataUnavailable
    case bluetoothUnavailable
    case motionUnavailable
    case modelLoadFailed
    case predictionFailed
    case pressureSensorError(String)
    case unknown(String)

    var id: String { localizedDescription }

    var errorDescription: String? {
        switch self {
        case .healthDataUnavailable: return "Health data is unavailable on this device."
        case .bluetoothUnavailable: return "Bluetooth is unavailable or powered off."
        case .motionUnavailable: return "Motion sensors are unavailable on this device."
        case .modelLoadFailed: return "Failed to load the Core ML model."
        case .predictionFailed: return "Prediction failed."
        case .pressureSensorError(let message): return "Pressure sensor error: \(message)"
        case .unknown(let message): return message
        }
    }
}

struct ErrorBanner: View {
    let error: AppError

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
            Text(error.localizedDescription).font(.callout)
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .shadow(radius: 5)
    }
}

extension View {
    func errorOverlay(_ error: AppError?) -> some View {
        ZStack(alignment: .top) {
            self
            if let error = error {
                ErrorBanner(error: error)
                    .padding()
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }
}