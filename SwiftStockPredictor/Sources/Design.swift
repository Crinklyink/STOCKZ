import SwiftUI

enum Design {
    static let tint = Color.accentColor
    static let blue = Color(red: 0.23, green: 0.50, blue: 0.96)
    static let green = Color(red: 0.12, green: 0.68, blue: 0.42)
    static let yellow = Color(red: 0.96, green: 0.65, blue: 0.16)
    static let red = Color(red: 0.92, green: 0.28, blue: 0.30)
    static let purple = Color(red: 0.54, green: 0.42, blue: 0.90)
    static let teal = Color(red: 0.10, green: 0.66, blue: 0.72)
    static let ink = Color.primary
    static let secondary = Color.secondary
    static let radius: CGFloat = 8
    static let pageMaxWidth: CGFloat = 1240
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
            .background(panelTint, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: Design.radius, style: .continuous)
                    .stroke(borderTint, lineWidth: 1)
            }
    }

    private var panelTint: Color {
        colorScheme == .dark ? Color(red: 0.08, green: 0.105, blue: 0.125).opacity(0.92) : .white.opacity(0.76)
    }

    private var borderTint: Color {
        colorScheme == .dark ? .white.opacity(0.13) : .white.opacity(0.72)
    }
}

struct PageFrame<Content: View>: View {
    let content: Content
    var maxWidth: CGFloat = Design.pageMaxWidth

    init(maxWidth: CGFloat = Design.pageMaxWidth, @ViewBuilder content: () -> Content) {
        self.maxWidth = maxWidth
        self.content = content()
    }

    var body: some View {
        ScrollView {
            content
                .frame(maxWidth: maxWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.horizontal, 4)
                .padding(.bottom, 28)
        }
        .scrollIndicators(.visible)
    }
}

struct MetricTile: View {
    @Environment(\.colorScheme) private var colorScheme
    let title: String
    let value: String
    let footnote: String
    var tone: Color = .primary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(value)
                .font(.system(size: 25, weight: .semibold, design: .rounded))
                .foregroundStyle(tone)
                .lineLimit(2)
                .minimumScaleFactor(0.62)
            Text(footnote)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(minHeight: 92, alignment: .topLeading)
        .padding(14)
        .background(tileTint, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Design.radius, style: .continuous)
                .stroke(.secondary.opacity(colorScheme == .dark ? 0.16 : 0.12), lineWidth: 1)
        }
    }

    private var tileTint: Color {
        colorScheme == .dark ? .white.opacity(0.045) : .white.opacity(0.62)
    }
}

struct Pill: View {
    let text: String
    var color: Color = .accentColor

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(color.opacity(0.14), in: Capsule())
            .foregroundStyle(color)
            .lineLimit(1)
            .minimumScaleFactor(0.7)
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
                .font(.system(size: 27, weight: .semibold, design: .rounded))
            if let subtitle {
                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
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
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .font(.headline.weight(.semibold))
                .foregroundStyle(color)
                .frame(width: 30, height: 30)
                .background(color.opacity(0.15), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            Spacer()
        }
        .padding(15)
        .background(color.opacity(0.055), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
        .background(Color(red: 0.08, green: 0.105, blue: 0.125).opacity(0.88), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Design.radius, style: .continuous)
                .stroke(color.opacity(0.20), lineWidth: 1)
        }
    }
}

struct InfoRow: View {
    let title: String
    let value: String
    var symbol: String = "circle"
    var tone: Color = .primary

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(tone)
                .frame(width: 22)
            Text(title)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .foregroundStyle(.primary)
        }
        .font(.callout)
    }
}

struct DataRow: View {
    let title: String
    let value: String
    var tone: Color = .primary

    var body: some View {
        HStack {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .font(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(tone)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
    }
}

struct StepPill: View {
    let number: Int
    let title: String
    let detail: String
    var color: Color = Design.blue

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)")
                .font(.caption.weight(.bold))
                .frame(width: 22, height: 22)
                .background(color.opacity(0.16), in: Circle())
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.callout.weight(.semibold))
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }
}

struct MiniBar: View {
    let value: Double
    var tint: Color = .blue

    var body: some View {
        ZStack(alignment: .leading) {
            Capsule().fill(.secondary.opacity(0.16))
            Capsule()
                .fill(tint)
                .scaleEffect(x: min(max(value, 0), 1), y: 1, anchor: .leading)
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

    var marketCapText: String {
        if self >= 1_000_000_000_000 { return String(format: "$%.1fT", self / 1_000_000_000_000.0) }
        if self >= 1_000_000_000 { return String(format: "$%.1fB", self / 1_000_000_000.0) }
        if self >= 1_000_000 { return String(format: "$%.1fM", self / 1_000_000.0) }
        return moneyText
    }
}

extension String {
    var cleanedProfile: String {
        replacingOccurrences(of: "_", with: " ").capitalized
    }
}

extension Int {
    var compactText: String {
        if self >= 1_000_000 { return String(format: "%.1fM", Double(self) / 1_000_000.0) }
        if self >= 1_000 { return String(format: "%.1fk", Double(self) / 1_000.0) }
        return "\(self)"
    }
}
