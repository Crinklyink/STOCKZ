import SwiftUI

enum Design {
    static let tint = Color.accentColor
    static let green = Color(red: 0.13, green: 0.66, blue: 0.36)
    static let yellow = Color(red: 0.92, green: 0.58, blue: 0.08)
    static let red = Color(red: 0.86, green: 0.23, blue: 0.25)
    static let ink = Color.primary
    static let secondary = Color.secondary
}

struct GlassPanel<Content: View>: View {
    @Environment(\.colorScheme) private var colorScheme
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(18)
            .background(colorScheme == .dark ? .regularMaterial : .thinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .background(panelTint, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(borderTint, lineWidth: 1)
            }
            .shadow(color: shadowTint, radius: 24, y: 14)
    }

    private var panelTint: Color {
        colorScheme == .dark ? .white.opacity(0.035) : .white.opacity(0.42)
    }

    private var borderTint: Color {
        colorScheme == .dark ? .white.opacity(0.12) : .white.opacity(0.62)
    }

    private var shadowTint: Color {
        colorScheme == .dark ? .black.opacity(0.22) : .blue.opacity(0.08)
    }
}

struct MetricTile: View {
    let title: String
    let value: String
    let footnote: String
    var tone: Color = .primary

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 10) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                    .foregroundStyle(tone)
                    .lineLimit(2)
                    .minimumScaleFactor(0.56)
                Text(footnote)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct Pill: View {
    let text: String
    var color: Color = .accentColor

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.14), in: Capsule())
            .foregroundStyle(color)
    }
}

struct SectionHeader: View {
    let title: String
    let subtitle: String?

    init(_ title: String, subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.title2.weight(.semibold))
            if let subtitle {
                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct StatusStrip: View {
    let title: String
    let detail: String
    var symbol: String = "sparkles"
    var color: Color = .blue

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol)
                .font(.title3.weight(.semibold))
                .foregroundStyle(color)
                .frame(width: 34, height: 34)
                .background(color.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer()
        }
        .padding(14)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(.white.opacity(0.18), lineWidth: 1)
        }
    }
}

struct MiniBar: View {
    let value: Double
    var tint: Color = .blue

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(.secondary.opacity(0.16))
                Capsule()
                    .fill(tint)
                    .frame(width: max(8, proxy.size.width * min(max(value, 0), 1)))
            }
        }
        .frame(height: 7)
    }
}

extension Double {
    var percentText: String {
        String(format: "%+.1f%%", self)
    }

    var moneyText: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: self)) ?? "$0.00"
    }
}

extension String {
    var cleanedProfile: String {
        replacingOccurrences(of: "_", with: " ")
    }
}
