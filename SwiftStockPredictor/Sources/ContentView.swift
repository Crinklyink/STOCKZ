import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.colorScheme) private var colorScheme
    @State private var section: AppSection = .dashboard

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 280)
        } detail: {
            ZStack {
                MeshBackground()
                detailView
                    .padding(24)
            }
        }
        .tint(.blue)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Stock Predictor")
                    .font(.title2.weight(.semibold))
                Text(appState.currentJob)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 12)

            ForEach(AppSection.allCases) { item in
                Button {
                    section = item
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: item.symbol)
                            .frame(width: 18)
                        Text(item.rawValue)
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, minHeight: 38, alignment: .leading)
                    .padding(.horizontal, 10)
                    .background(section == item ? Color.accentColor.opacity(colorScheme == .dark ? 0.22 : 0.14) : Color.clear, in: RoundedRectangle(cornerRadius: 10))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(section == item ? Color.accentColor : Color.primary)
            }

            Spacer()

            GlassPanel {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: appState.isRunning ? "bolt.horizontal.circle.fill" : "checkmark.seal.fill")
                            .foregroundStyle(appState.isRunning ? .blue : Design.green)
                        Text("Model")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                    Text(appState.modelMetadata?.stackName ?? "Not trained")
                        .font(.callout.weight(.semibold))
                        .lineLimit(2)
                    HStack {
                        Text(String(format: "AUC %.3f", appState.modelMetadata?.activeAuc ?? 0))
                        Spacer()
                        Text("\(appState.modelMetadata?.trainingSamples ?? 0) rows")
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

    private var sidebarBackground: some View {
        Rectangle()
            .fill(colorScheme == .dark ? Color.black.opacity(0.16) : Color.white.opacity(0.34))
            .background(.ultraThinMaterial)
    }

    @ViewBuilder
    private var detailView: some View {
        switch section {
        case .dashboard:
            DashboardScreen()
        case .picks:
            PicksScreen()
        case .training:
            TrainingScreen(mode: .training)
        case .backtesting:
            TrainingScreen(mode: .backtesting)
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
            Color.blue.opacity(0.12),
            Color.cyan.opacity(0.08),
            Color(red: 0.99, green: 0.99, blue: 1.0)
        ]
    }

    private var darkColors: [Color] {
        [
            Color(red: 0.05, green: 0.07, blue: 0.10),
            Color.blue.opacity(0.20),
            Color.cyan.opacity(0.10),
            Color(red: 0.02, green: 0.025, blue: 0.035)
        ]
    }
}
