import SwiftUI

struct DashboardScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let compact = width < 860
            let wide = width >= 1110
            PageFrame(maxWidth: wide ? 1360 : .infinity) {
                VStack(alignment: .leading, spacing: compact ? 14 : 18) {
                    DashboardHero(compact: compact)

                    DashboardKPIBar(compact: width < 1000)

                    if wide {
                        HStack(alignment: .top, spacing: 18) {
                            WatchlistTable(compact: false)
                                .frame(maxWidth: .infinity)
                            DashboardRightRail()
                                .frame(width: 340)
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 14) {
                            WatchlistTable(compact: true)
                            DashboardRightRail()
                        }
                    }
                }
            }
        }
    }
}

struct DashboardHero: View {
    @EnvironmentObject private var appState: AppState
    var compact: Bool = false

    var body: some View {
        HStack(alignment: compact ? .center : .top) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Good morning, Trader")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text("Dashboard")
                    .font(.system(size: compact ? 32 : 38, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                Text("AI-powered picks, model sync, regime detection, and paper-trade state in one place.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer()
            if !compact {
                HStack(spacing: 14) {
                    Image(systemName: "clock")
                        .font(.title2.weight(.medium))
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Last updated")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(appState.scanGeneratedText)
                            .font(.callout.monospacedDigit().weight(.semibold))
                    }
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 14)
                .background(Color(red: 0.08, green: 0.105, blue: 0.125).opacity(0.88), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: Design.radius, style: .continuous)
                        .stroke(.white.opacity(0.16), lineWidth: 1)
                }
            }
        }
    }
}

struct DashboardKPIBar: View {
    @EnvironmentObject private var appState: AppState
    var compact: Bool = false

    var body: some View {
        GlassPanel {
            if compact {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 12)], spacing: 12) {
                    kpis
                }
            } else {
                HStack(spacing: 0) {
                    kpisWithSeparators
                }
            }
        }
    }

    @ViewBuilder
    private var kpis: some View {
        DashboardKPI(
            title: "This Week",
            value: "\(appState.candidates.count)",
            subtitle: "ranked names",
            color: Design.blue,
            samples: [0.12, 0.15, 0.14, 0.18, 0.20, 0.33, 0.30, 0.42, 0.61, 0.66]
        )
        DashboardKPI(
            title: "Model AUC",
            value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0),
            subtitle: appState.modelMetadata?.stackName ?? "model stack",
            color: Design.green,
            samples: [0.12, 0.14, 0.18, 0.20, 0.28, 0.34, 0.49, 0.58, 0.60, 0.72]
        )
        DashboardKPI(
            title: "VIX",
            value: String(format: "%.1f", appState.summary?.vix ?? 0),
            subtitle: (appState.summary?.regime ?? "market regime").replacingOccurrences(of: "_", with: " "),
            color: Design.teal,
            samples: [0.16, 0.18, 0.16, 0.20, 0.22, 0.39, 0.36, 0.55, 0.52, 0.65]
        )
        DashboardKPI(
            title: "Portfolio",
            value: (appState.modelMetadata?.trainingSamples ?? 0).compactText,
            subtitle: appState.modelMetadata?.trainedDateText ?? "training rows",
            color: Design.purple,
            samples: [0.20, 0.22, 0.28, 0.32, 0.45, 0.55, 0.38, 0.62, 0.50, 0.74],
            style: .bars
        )
    }

    @ViewBuilder
    private var kpisWithSeparators: some View {
        DashboardKPI(
            title: "This Week",
            value: "\(appState.candidates.count)",
            subtitle: "ranked names",
            color: Design.blue,
            samples: [0.12, 0.15, 0.14, 0.18, 0.20, 0.33, 0.30, 0.42, 0.61, 0.66]
        )
        KPISeparator()
        DashboardKPI(
            title: "Model AUC",
            value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0),
            subtitle: appState.modelMetadata?.stackName ?? "model stack",
            color: Design.green,
            samples: [0.12, 0.14, 0.18, 0.20, 0.28, 0.34, 0.49, 0.58, 0.60, 0.72]
        )
        KPISeparator()
        DashboardKPI(
            title: "VIX",
            value: String(format: "%.1f", appState.summary?.vix ?? 0),
            subtitle: (appState.summary?.regime ?? "market regime").replacingOccurrences(of: "_", with: " "),
            color: Design.teal,
            samples: [0.16, 0.18, 0.16, 0.20, 0.22, 0.39, 0.36, 0.55, 0.52, 0.65]
        )
        KPISeparator()
        DashboardKPI(
            title: "Portfolio",
            value: (appState.modelMetadata?.trainingSamples ?? 0).compactText,
            subtitle: appState.modelMetadata?.trainedDateText ?? "training rows",
            color: Design.purple,
            samples: [0.20, 0.22, 0.28, 0.32, 0.45, 0.55, 0.38, 0.62, 0.50, 0.74],
            style: .bars
        )
    }
}

struct DashboardKPI: View {
    enum ChartStyle { case line, bars }

    let title: String
    let value: String
    let subtitle: String
    let color: Color
    let samples: [Double]
    var style: ChartStyle = .line

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(value)
                .font(.system(size: 30, weight: .semibold, design: .rounded))
                .monospacedDigit()
            HStack(alignment: .bottom, spacing: 12) {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                if style == .line {
                    Sparkline(samples: samples, color: color)
                        .frame(width: 92, height: 42)
                } else {
                    MiniBars(samples: samples, color: color)
                        .frame(width: 92, height: 42)
                }
            }
        }
        .frame(maxWidth: .infinity, minHeight: 108, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.white.opacity(0.035), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
    }
}

struct KPISeparator: View {
    var body: some View {
        Rectangle()
            .fill(.white.opacity(0.12))
            .frame(width: 1, height: 86)
            .padding(.horizontal, 10)
    }
}

struct WatchlistTable: View {
    @EnvironmentObject private var appState: AppState
    var compact: Bool = false
    @State private var detailCandidate: Candidate?

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Label("Watchlist", systemImage: "waveform")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Pill(text: "\(min(appState.candidates.count, 10)) ranked", color: .secondary)
                }
                VStack(spacing: 6) {
                    if !compact {
                        WatchlistHeader()
                    }
                    ForEach(Array(appState.candidates.prefix(10).enumerated()), id: \.element.id) { index, candidate in
                        Button {
                            detailCandidate = candidate
                        } label: {
                            DashboardCandidateRow(candidate: candidate, rank: index + 1, compact: compact)
                        }
                        .buttonStyle(.plain)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                }
            }
        }
        .sheet(item: $detailCandidate) { candidate in
            PickDetailDrilldown(candidate: candidate)
                .environmentObject(appState)
                .frame(minWidth: 760, minHeight: 720)
        }
    }
}

struct WatchlistHeader: View {
    var body: some View {
        HStack(spacing: 14) {
            TableHead("#")
                .frame(width: 38)
            TableHead("Ticker")
                .frame(width: 74, alignment: .leading)
            TableHead("Company")
            Spacer()
            TableHead("Score")
                .frame(width: 70, alignment: .trailing)
            TableHead("Confidence")
                .frame(width: 150, alignment: .trailing)
        }
        .padding(.horizontal, 10)
        .padding(.bottom, 4)
    }
}

struct TableHead: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }
}

struct DashboardCandidateRow: View {
    let candidate: Candidate
    let rank: Int
    var compact: Bool = false

    var body: some View {
        if compact {
            compactRow
        } else {
            wideRow
        }
    }

    private var wideRow: some View {
        HStack(spacing: 14) {
            Text("\(rank)")
                .font(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(.primary)
                .frame(width: 28, height: 28)
                .background(.white.opacity(0.10), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                .frame(width: 38)

            Text(candidate.ticker)
                .font(.callout.weight(.bold))
                .lineLimit(1)
                .frame(width: 74, alignment: .leading)

            Text(candidate.companyName ?? candidate.sector ?? "Unknown")
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .frame(maxWidth: .infinity, alignment: .leading)

            Text(String(format: "%.1f", candidate.finalScore ?? 0))
                .font(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(scoreColor)
                .frame(width: 70, alignment: .trailing)

            HStack(spacing: 10) {
                Text(String(format: "%.0f%%", candidate.confidencePercent))
                    .font(.callout.monospacedDigit().weight(.semibold))
                    .foregroundStyle(Design.green)
                    .frame(width: 44, alignment: .trailing)
                MiniBar(value: min(max(candidate.confidencePercent / 100.0, 0), 1), tint: Design.green)
                    .frame(width: 86, height: 8)
            }
            .frame(width: 150, alignment: .trailing)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(rowTint, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
    }

    private var compactRow: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 10) {
                Text("\(rank)")
                    .font(.caption.monospacedDigit().weight(.bold))
                    .frame(width: 26, height: 26)
                    .background(.white.opacity(0.10), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(candidate.ticker)
                        .font(.headline.weight(.bold))
                    Text(candidate.companyName ?? candidate.sector ?? "Unknown")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Text(String(format: "%.1f", candidate.finalScore ?? 0))
                    .font(.headline.monospacedDigit().weight(.semibold))
                    .foregroundStyle(scoreColor)
            }
            HStack(spacing: 10) {
                Text(String(format: "%.0f%% confidence", candidate.confidencePercent))
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundStyle(Design.green)
                MiniBar(value: min(max(candidate.confidencePercent / 100.0, 0), 1), tint: Design.green)
                    .frame(height: 8)
            }
        }
        .padding(10)
        .background(rowTint, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
    }

    private var scoreColor: Color {
        let score = candidate.finalScore ?? 0
        if score >= 65 { return Design.green }
        if score >= 50 { return Design.blue }
        return .primary
    }

    private var rowTint: Color {
        rank <= 3 ? .white.opacity(0.065) : .white.opacity(0.035)
    }
}

struct DashboardRightRail: View {
    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 14) {
                ActiveModelCard()
                    .frame(minWidth: 250)
                MarketCard()
                    .frame(minWidth: 250)
                RecentScansCard()
                    .frame(minWidth: 250)
            }
            VStack(spacing: 14) {
                ActiveModelCard()
                MarketCard()
                RecentScansCard()
            }
        }
    }
}

struct ActiveModelCard: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text("Active Model")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Pill(text: appState.scanUsesOlderModel ? "Refresh" : "Synced", color: appState.scanUsesOlderModel ? Design.yellow : Design.green)
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 18) {
                        StackedModelIcon()
                            .frame(width: 104, height: 116)
                        Rectangle()
                            .fill(.white.opacity(0.12))
                            .frame(width: 1, height: 112)
                        modelDetails
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            StackedModelIcon()
                                .frame(width: 82, height: 74)
                            Spacer()
                        }
                        modelDetails
                    }
                }
            }
        }
    }

    private var modelDetails: some View {
        VStack(alignment: .leading, spacing: 12) {
            DataLabel(title: "Model", value: appState.modelMetadata?.stackName ?? "--")
            DataLabel(title: "Target", value: appState.modelMetadata?.targetSummary ?? "--")
            DataLabel(title: "Regime", value: appState.latestScan?.regimeLabel?.lowercased() ?? appState.summary?.regime ?? "--", tone: Design.blue)
            DataLabel(title: "Last scan", value: appState.scanGeneratedText)
        }
    }
}

struct DataLabel: View {
    let title: String
    let value: String
    var tone: Color = .primary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.weight(.semibold))
                .foregroundStyle(tone)
                .lineLimit(2)
                .minimumScaleFactor(0.7)
        }
    }
}

struct MarketCard: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 16) {
                Text("Market")
                    .font(.title3.weight(.semibold))
                GaugeArc(value: appState.summary?.vix ?? 0)
                    .frame(height: 132)
                Divider().opacity(0.35)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 90), spacing: 12)], spacing: 12) {
                    DataLabel(title: "SPY Week", value: (appState.summary?.spyWeekReturn ?? 0).percentText, tone: Design.green)
                    DataLabel(title: "S&P 500", value: "5,123.4", tone: Design.blue)
                    DataLabel(title: "Trend", value: "Slightly Bullish", tone: Design.green)
                }
            }
        }
    }
}

struct RecentScansCard: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 14) {
                Text("Recent Scans")
                    .font(.title3.weight(.semibold))
                ForEach(scanRows, id: \.date) { row in
                    HStack(spacing: 10) {
                        Image(systemName: "checkmark.circle")
                            .foregroundStyle(Design.green)
                        Text(row.date)
                            .font(.caption.monospacedDigit())
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                        Spacer()
                        Text(row.regime)
                            .font(.caption)
                            .foregroundStyle(Design.blue)
                            .lineLimit(1)
                        Text(row.samples)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }
                }
                Divider().opacity(0.35)
                HStack {
                    Text("View full history")
                        .font(.callout)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var scanRows: [(date: String, regime: String, samples: String)] {
        let current = appState.scanGeneratedText
        let regime = appState.latestScan?.regimeLabel?.lowercased() ?? "risk_on"
        let samples = "\((appState.modelMetadata?.trainingSamples ?? 0).compactText) samples"
        return [
            (current, regime, samples),
            ("2026-04-20 01:42", "risk_on", "54.1k samples"),
            ("2026-04-13 01:41", "risk_neutral", "53.8k samples"),
            ("2026-04-06 01:42", "risk_off", "54.3k samples"),
        ]
    }
}

struct StackedModelIcon: View {
    var body: some View {
        ZStack {
            ForEach(0..<4, id: \.self) { index in
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(layerGradient(index))
                    .frame(width: 76, height: 42)
                    .rotationEffect(.degrees(45))
                    .offset(y: CGFloat(index * 18 - 26))
                    .opacity(0.92)
            }
        }
    }

    private func layerColor(_ index: Int) -> Color {
        [Design.blue, Design.teal, Design.green, Design.purple][index]
    }

    private func layerGradient(_ index: Int) -> LinearGradient {
        LinearGradient(
            colors: [layerColor(index).opacity(0.85), layerColor(index).opacity(0.28)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}

struct Sparkline: View {
    let samples: [Double]
    let color: Color

    var body: some View {
        GeometryReader { proxy in
            Path { path in
                guard let first = samples.first else { return }
                let width = proxy.size.width
                let height = proxy.size.height
                let step = width / CGFloat(max(samples.count - 1, 1))
                path.move(to: CGPoint(x: 0, y: height * (1 - first)))
                for index in samples.indices {
                    path.addLine(to: CGPoint(x: CGFloat(index) * step, y: height * (1 - samples[index])))
                }
            }
            .stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
    }
}

struct MiniBars: View {
    let samples: [Double]
    let color: Color

    var body: some View {
        HStack(alignment: .bottom, spacing: 5) {
            ForEach(samples.indices, id: \.self) { index in
                Capsule()
                    .fill(color.opacity(0.32 + 0.055 * Double(index)))
                    .frame(width: 4, height: max(4, 38 * samples[index]))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
    }
}

struct GaugeArc: View {
    let value: Double

    var body: some View {
        ZStack {
            GaugeArcShape(progress: 1)
                .stroke(.white.opacity(0.10), style: StrokeStyle(lineWidth: 12, lineCap: .round))
            GaugeArcShape(progress: min(max((value - 10) / 20, 0), 1))
                .stroke(
                    LinearGradient(colors: [Design.green, Design.yellow, Design.red, Design.purple], startPoint: .leading, endPoint: .trailing),
                    style: StrokeStyle(lineWidth: 12, lineCap: .round)
                )
            VStack(spacing: 2) {
                Text(String(format: "%.1f", value))
                    .font(.system(size: 34, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                Text("VIX")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("Caution")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .offset(y: 18)
            HStack {
                Text("10\nCalm")
                    .multilineTextAlignment(.center)
                Spacer()
                Text("30\nRisk")
                    .multilineTextAlignment(.center)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .offset(y: 48)
        }
    }
}

struct GaugeArcShape: Shape {
    var progress: Double

    var animatableData: Double {
        get { progress }
        set { progress = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.maxY - 12)
        let radius = min(rect.width * 0.42, rect.height * 0.92)
        path.addArc(
            center: center,
            radius: radius,
            startAngle: .degrees(190),
            endAngle: .degrees(190 + 160 * progress),
            clockwise: false
        )
        return path
    }
}

struct RegimeGauge: View {
    let value: Double

    var body: some View {
        GaugeArc(value: value)
            .frame(height: 130)
    }
}

struct EmptyState: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 160)
    }
}
