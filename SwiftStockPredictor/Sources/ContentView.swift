import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.colorScheme) private var colorScheme
    @State private var section: AppSection = .dashboard

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 230, ideal: 260, max: 300)
        } detail: {
            ZStack {
                MeshBackground()
                GeometryReader { proxy in
                    detailView
                        .padding(proxy.size.width < 900 ? 14 : 22)
                }
            }
        }
        .tint(.blue)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(width: 32, height: 32)
                        .background(Design.blue, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Stock Predictor")
                            .font(.headline.weight(.semibold))
                        Text("Weekly model lab")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if appState.isRunning {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.62)
                        Text("\(appState.currentJob) | \(appState.elapsedSeconds)s")
                            .font(.caption)
                            .foregroundStyle(Design.blue)
                            .monospacedDigit()
                    }
                } else {
                    Pill(text: appState.currentJob, color: .secondary)
                }
            }
            .padding(.top, 10)
            .padding(.bottom, 6)

            ForEach(AppSection.allCases) { item in
                Button {
                    section = item
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: item.symbol)
                            .frame(width: 18)
                        Text(item.rawValue)
                            .font(.callout.weight(section == item ? .semibold : .regular))
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                    .padding(.horizontal, 12)
                    .background(section == item ? selectedNavTint : Color.clear, in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                    .overlay(alignment: .leading) {
                        if section == item {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Design.blue)
                                .frame(width: 3, height: 20)
                                .padding(.leading, 2)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .buttonStyle(.plain)
                .foregroundStyle(section == item ? Color.accentColor : Color.primary)
            }

            Spacer()

            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: appState.isRunning ? "bolt.horizontal.circle.fill" : "checkmark.seal.fill")
                            .foregroundStyle(appState.isRunning ? Design.blue : Design.green)
                        Text(appState.scanUsesOlderModel ? "Refresh scan" : "In sync")
                            .font(.caption.weight(.semibold))
                        Spacer()
                        Pill(
                            text: appState.scanUsesOlderModel ? "Refresh" : "Synced",
                            color: appState.scanUsesOlderModel ? Design.yellow : Design.green
                        )
                    }
                    Text(appState.modelMetadata?.stackName ?? "Not trained")
                        .font(.title3.weight(.semibold))
                        .lineLimit(2)
                    Text(appState.modelMetadata?.targetSummary ?? "weekly target")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    HStack {
                        Text(String(format: "AUC %.3f", appState.modelMetadata?.activeAuc ?? 0))
                        Spacer()
                        Text("\((appState.modelMetadata?.trainingSamples ?? 0).compactText) rows")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    MiniBar(value: min(max(appState.modelMetadata?.activeAuc ?? 0, 0), 1), tint: (appState.modelMetadata?.activeAuc ?? 0) >= 0.6 ? Design.green : Design.yellow)
                    Button {
                        appState.runScan()
                    } label: {
                        Label("Run Scan", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(appState.isRunning)
                }
            }
        }
        .padding(18)
        .background(sidebarBackground)
    }

    private var selectedNavTint: Color {
        colorScheme == .dark ? Design.blue.opacity(0.22) : Design.blue.opacity(0.13)
    }

    private var sidebarBackground: some View {
        Rectangle()
            .fill(colorScheme == .dark ? Color(red: 0.055, green: 0.075, blue: 0.095).opacity(0.98) : Color.white.opacity(0.88))
    }

    @ViewBuilder
    private var detailView: some View {
        switch section {
        case .dashboard:
            DashboardScreen()
        case .picks:
            PicksScreen()
        case .risk:
            RiskDashboardScreen()
        case .alerts:
            AlertsScreen()
        case .training:
            TrainingScreen(mode: .training)
        case .backtesting:
            TrainingScreen(mode: .backtesting)
        case .modelLab:
            ModelLabScreen()
        case .history:
            HistoryScreen()
        case .settings:
            SettingsScreen()
        }
    }
}

struct MeshBackground: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        LinearGradient(
            colors: colorScheme == .dark ? darkColors : lightColors,
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    private var lightColors: [Color] {
        [
            Color(red: 0.96, green: 0.98, blue: 1.0),
            Color(red: 0.91, green: 0.96, blue: 0.98),
            Color(red: 0.98, green: 0.98, blue: 1.0)
        ]
    }

    private var darkColors: [Color] {
        [
            Color(red: 0.045, green: 0.065, blue: 0.085),
            Color(red: 0.055, green: 0.10, blue: 0.12),
            Color(red: 0.025, green: 0.03, blue: 0.04)
        ]
    }
}
