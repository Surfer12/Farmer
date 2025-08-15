import Foundation
import CoreGraphics

enum PressureColorMap {
    static func cgImage(width: Int, height: Int, normalized: [Float]) -> CGImage? {
        guard width > 0, height > 0, normalized.count == width * height else { return nil }
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        for j in 0..<height {
            for i in 0..<width {
                let idx = j * width + i
                let v = max(0.0, min(1.0, normalized[idx]))
                let color = colorRGBA(for: v)
                let o = idx * bytesPerPixel
                buffer[o + 0] = color.r
                buffer[o + 1] = color.g
                buffer[o + 2] = color.b
                buffer[o + 3] = 255
            }
        }

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return buffer.withUnsafeMutableBytes { ptr -> CGImage? in
            guard let context = CGContext(data: ptr.baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow,
                                          space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            return context.makeImage()
        }
    }

    private static func colorRGBA(for value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let v = max(0.0, min(1.0, value))
        let c: (Double, Double, Double)
        if v < 0.25 {
            let t = Double(v / 0.25)
            c = (0.0, t, 1.0)
        } else if v < 0.5 {
            let t = Double((v - 0.25) / 0.25)
            c = (0.0, 1.0, 1.0 - t)
        } else if v < 0.75 {
            let t = Double((v - 0.5) / 0.25)
            c = (t, 1.0, 0.0)
        } else {
            let t = Double((v - 0.75) / 0.25)
            c = (1.0, 1.0 - t, 0.0)
        }
        return (UInt8(c.0 * 255.0), UInt8(c.1 * 255.0), UInt8(c.2 * 255.0))
    }
}
