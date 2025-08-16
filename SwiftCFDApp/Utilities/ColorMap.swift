import SwiftUI
import SceneKit

struct ColorMap {
    static func pressureColor(_ value: Float) -> PlatformColor {
        let clamped = max(0, min(1, CGFloat(value)))
        // Blue → Red gradient with slight gamma
        let gamma: CGFloat = 0.8
        let red = pow(clamped, gamma)
        let blue = pow(1 - clamped, gamma)
        return PlatformColor.rgb(red, 0.1, blue, 1)
    }

    static func image(from values: [Float], width: Int = 64, height: Int = 64) -> PlatformImage {
        let count = max(1, min(values.count, width * height))
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<count {
            let color = pressureColor(values[i])
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            color.getRed(&r, green: &g, blue: &b, alpha: &a)
            let idx = i * 4
            pixels[idx + 0] = UInt8(r * 255)
            pixels[idx + 1] = UInt8(g * 255)
            pixels[idx + 2] = UInt8(b * 255)
            pixels[idx + 3] = 255
        }
        let provider = CGDataProvider(data: NSData(bytes: &pixels, length: pixels.count))!
        let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
        #if os(macOS)
        return PlatformImage(cgImage: cgImage, width: width, height: height)
        #else
        return PlatformImage(cgImage: cgImage)
        #endif
    }
}