# =============================================================================
# US GDP Nowcast Dashboard — Shiny for Python
# Uses real New York Fed Staff Nowcast data (2002–2021) from an Excel file.
#
# Data source: New York Fed Staff Nowcast
#   https://www.newyorkfed.org/research/policy/nowcast
#
# The Excel workbook contains two relevant sheets:
#   1. "Forecasts By Quarter" — wide format: each column is a target quarter
#      (e.g. 2002Q1 … 2021Q3), each row is a weekly forecast date, and the
#      cell value is the GDP-growth nowcast issued on that date for that quarter.
#      → We use this sheet for the MAIN nowcast evolution line.
#
#   2. "Forecasts By Horizon" — long format with columns:
#        Forecast date | Reference quarter | Backcast | Nowcast | Forecast
#      → We use the Backcast (previous-quarter revision) and Forecast
#        (next-quarter lookahead) columns as optional comparison overlays.
#
# Shiny for Python equivalents of R Shiny concepts:
#   R Shiny                →  Shiny for Python
#   -------                →  ----------------
#   fluidPage()            →  ui.page_fluid()
#   selectInput()          →  ui.input_select()
#   checkboxInput()        →  ui.input_checkbox()
#   checkboxGroupInput()   →  ui.input_checkbox_group()
#   conditionalPanel()     →  ui.panel_conditional()
#   plotlyOutput()         →  output_widget()              (from shinywidgets)
#   renderPlotly()         →  @render_plotly                (from shinywidgets)
#   reactive()             →  @reactive.calc
#   observe()              →  @reactive.effect
#   updateSelectInput()    →  ui.update_select()
# =============================================================================

# --- Core Shiny framework ---
from shiny import App, ui, reactive         # App framework, UI builders, reactivity
from shinywidgets import output_widget, render_plotly  # Plotly ↔ Shiny bridge

# --- Plotting ---
import plotly.graph_objects as go            # Low-level Plotly figure construction
import plotly.express as px                 # High-level Plotly interface

# --- Data handling ---
import pandas as pd


# =============================================================================
# DATA LOADING
# Read both sheets once at startup and reshape into easy-to-query structures.
# =============================================================================

EXCEL_PATH = "data/New-York-Fed-Staff-Nowcast_data_2002-2021.xlsx"

# ---- Sheet 1: "Forecasts By Quarter" (wide format) -------------------------
# Row 13 (0-indexed) is the header row with "Forecast Date", "2002Q1", …
# Each subsequent row is one weekly forecast vintage.
df_by_quarter = pd.read_excel(EXCEL_PATH, sheet_name="Forecasts By Quarter", header=13)
df_by_quarter.rename(columns={df_by_quarter.columns[0]: "forecast_date"}, inplace=True)
df_by_quarter["forecast_date"] = pd.to_datetime(df_by_quarter["forecast_date"])

# Build the list of available quarters from the column headers (all except the date col).
all_quarters = [c for c in df_by_quarter.columns if c != "forecast_date"]
# e.g. ["2002Q1", "2002Q2", …, "2021Q3"]

# ---- Sheet 2: "Forecasts By Horizon" (long format) -------------------------
# Same header-row offset. Columns are:
#   Forecast date | Reference quarter | Backcast | Nowcast | Forecast
df_by_horizon = pd.read_excel(EXCEL_PATH, sheet_name="Forecasts By Horizon", header=13)
# Rename messy multi-line headers to clean names
df_by_horizon.columns = ["forecast_date", "ref_quarter", "backcast", "nowcast", "forecast"]
df_by_horizon["forecast_date"] = pd.to_datetime(df_by_horizon["forecast_date"])

# Convert numeric columns (they may have been read as object due to mixed NaN)
for col in ["backcast", "nowcast", "forecast"]:
    df_by_horizon[col] = pd.to_numeric(df_by_horizon[col], errors="coerce")

# ---- Derive year / quarter lookup for the dropdowns -------------------------
# Parse "2002Q1" → (2002, "Q1") and group by year.
quarter_year_map: dict[int, list[str]] = {}  # { year: ["Q1", "Q2", …] }
for q in all_quarters:
    yr = int(q[:4])
    qtr = q[4:]  # e.g. "Q1"
    quarter_year_map.setdefault(yr, []).append(qtr)

available_years = sorted(quarter_year_map.keys())

# Build {label: value} dicts for ui.input_select (Shiny uses dicts, not lists of dicts)
year_choices = {str(y): str(y) for y in available_years}
default_year = str(available_years[-1])
default_quarters = sorted(quarter_year_map[available_years[-1]])
quarter_choices = {q: q for q in default_quarters}



# =============================================================================
# HELPER: extract the time-series for a chosen quarter
# =============================================================================
def get_quarter_data(year: int, quarter: str) -> dict:
    """
    Return a dict with:
      - 'nowcast_series' : DataFrame(forecast_date, nowcast) from the By-Quarter sheet
      - 'horizon_series' : DataFrame(forecast_date, backcast, nowcast, forecast)
                           from the By-Horizon sheet, filtered to this ref quarter
    """
    label = f"{year}{quarter}"  # e.g. "2020Q3"

    # --- From the wide "By Quarter" sheet: pull the column for this quarter ---
    if label in df_by_quarter.columns:
        sub = df_by_quarter[["forecast_date", label]].dropna(subset=[label]).copy()
        sub.rename(columns={label: "nowcast"}, inplace=True)
    else:
        sub = pd.DataFrame(columns=["forecast_date", "nowcast"])

    # --- From the "By Horizon" sheet: rows where ref_quarter matches -----------
    horizon = df_by_horizon[df_by_horizon["ref_quarter"] == label].copy()

    return {"nowcast_series": sub, "horizon_series": horizon}


# =============================================================================
# UI LAYOUT  (equivalent of R Shiny's `ui <- fluidPage(...)`)
#
# Structure:
#   page_fluid (full-width page with Bootstrap grid)
#   ├── Custom CSS (injected via ui.tags.style)
#   ├── Title + subtitle
#   └── row
#       ├── column(8) → Chart card with Plotly widget
#       └── column(4) → Sidebar controls
#           ├── Card: Year + Quarter selectors
#           ├── Card: "Compare Data" checkbox
#           └── Card: Comparison series checkboxes (conditionally visible)
#
# Shiny for Python uses Bootstrap's 12-column grid, just like R Shiny.
# ui.row() + ui.column(width) maps directly to R's fluidRow() + column().
# =============================================================================

app_ui = ui.page_fluid(

    # ---- Inject custom CSS (equivalent of R's tags$head(tags$style(...))) ----
    ui.tags.style("""
        body { background-color: #f6f8fb; }
        .app-title { font-size: 30px; font-weight: 700; margin-bottom: 6px; }
        .app-subtitle { color: #5f6b7a; margin-bottom: 18px; }
        .control-card {
            background: white;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .chart-card {
            background: white;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
    """),

    # ---- Page header ----
    ui.div("US GDP Nowcast", class_="app-title"),
    ui.div(
        "New York Fed Staff Nowcast data (2002–2021). "
        "Shows how the GDP growth estimate for each quarter evolved over time.",
        class_="app-subtitle",
    ),

    # ---- Main content: 12-column Bootstrap grid (8 + 4 split) ----
    ui.row(

        # ========== LEFT: Chart (8 of 12 columns) ==========
        # output_widget() is the shinywidgets equivalent of R's plotlyOutput().
        # It renders an interactive Plotly figure pushed from the server.
        ui.column(
            8,
            ui.div(
                output_widget("nowcast_plot", height="520px"),
                class_="chart-card",
            ),
        ),

        # ========== RIGHT: Sidebar controls (4 of 12 columns) ==========
        ui.column(
            4,

            # --- Card 1: Quarter selector ---
            # Two dropdowns side by side for Year and Quarter.
            # ui.input_select() is the Shiny-for-Python equivalent of R's selectInput().
            # The `choices` argument accepts a dict of {display_label: value}.
            ui.div(
                ui.h4("Select quarter"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            id="year",
                            label="Year",
                            choices=year_choices,       # {"2002": "2002", …, "2021": "2021"}
                            selected=default_year,      # most recent year
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            id="quarter",
                            label="Quarter",
                            choices=quarter_choices,     # updated dynamically in server
                            selected=default_quarters[-1],
                        ),
                    ),
                ),
                class_="control-card",
            ),

            # --- Card 2: Display options ---
            # ui.input_checkbox() maps to R's checkboxInput() — a single TRUE/FALSE toggle.
            # When checked, it reveals the comparison-series card below.
            ui.div(
                ui.h4("Display options"),
                ui.input_checkbox(
                    id="compare_data",
                    label="Compare Data",
                    value=False,
                ),
                class_="control-card",
            ),

            # --- Card 3: Comparison series (conditionally visible) ---
            # ui.panel_conditional() is the direct equivalent of R's conditionalPanel().
            # The `condition` is a JavaScript expression evaluated in the browser;
            # `input.compare_data` mirrors the Python input ID.
            #
            # ui.input_checkbox_group() maps to R's checkboxGroupInput() — lets the
            # user pick multiple items from a list.
            #
            # Backcast = the Fed's revision of LAST quarter's GDP
            #            (available during the first ~4 weeks of a new quarter)
            # Forecast = the Fed's lookahead to NEXT quarter's GDP
            #            (available during the last ~4 weeks of a quarter)
            ui.panel_conditional(
                "input.compare_data",     # JS expression: true when checkbox is checked
                ui.div(
                    ui.h4("Comparison series"),
                    ui.input_checkbox_group(
                        id="comparison_series",
                        label="Overlay series",
                        choices={
                            "backcast": "Backcast (previous quarter)",
                            "forecast": "Forecast (next quarter)",
                        },
                        selected=["backcast", "forecast"],
                    ),
                    class_="control-card",
                ),
            ),
        ),
    ),
)


# =============================================================================
# SERVER  (equivalent of R Shiny's `server <- function(input, output, session)`)
#
# In Shiny for Python the server is a plain function that receives `input`,
# `output`, and `session`.  Reactive logic uses decorators:
#   @reactive.calc       →  like R's reactive()   (cached computed value)
#   @reactive.effect     →  like R's observe()    (side-effect, e.g. update UI)
#   @render_plotly       →  like R's renderPlotly() (push a Plotly figure)
#   ui.update_select()   →  like R's updateSelectInput()
# =============================================================================

def server(input, output, session):

    # -----------------------------------------------------------------------
    # Effect 1: When the year changes, update the quarter dropdown
    #
    # Equivalent of R's:
    #   observeEvent(input$year, {
    #     updateSelectInput(session, "quarter", choices = …)
    #   })
    #
    # @reactive.effect runs whenever any reactive input it reads changes.
    # Here it reads input.year(), so it fires on every year change.
    # -----------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.year)       # only trigger when year changes
    def _update_quarters():
        yr = int(input.year())
        quarters = sorted(quarter_year_map.get(yr, []))
        choices = {q: q for q in quarters}
        # Update the quarter dropdown's choices and select the last one
        ui.update_select(
            "quarter",
            choices=choices,
            selected=quarters[-1] if quarters else None,
        )

    # -----------------------------------------------------------------------
    # Reactive calc: fetch data for the currently selected year + quarter
    #
    # Equivalent of R's:
    #   nowcast_data <- reactive({ get_nowcast_data(input$year, input$quarter) })
    #
    # @reactive.calc caches the result and only recomputes when its
    # dependencies (input.year, input.quarter) change.
    # -----------------------------------------------------------------------
    @reactive.calc
    def current_data():
        yr = int(input.year())
        qtr = input.quarter()
        if not qtr:
            return {"nowcast_series": pd.DataFrame(), "horizon_series": pd.DataFrame()}
        return get_quarter_data(yr, qtr)

    # -----------------------------------------------------------------------
    # Render: build and return the Plotly figure
    #
    # Equivalent of R's:
    #   output$nowcast_plot <- renderPlotly({ … })
    #
    # @render_plotly automatically re-runs whenever any input.xxx() it reads
    # changes, and pushes the new figure to the output_widget("nowcast_plot")
    # in the UI.
    # -----------------------------------------------------------------------
    @render_plotly
    def nowcast_plot():
        data = current_data()
        nowcast_df = data["nowcast_series"]
        horizon_df = data["horizon_series"]

        yr = input.year()
        qtr = input.quarter()

        # Start with an empty figure; traces are added conditionally

        # ---- Main line: NY Fed Nowcast for this quarter ----
        # This shows how the GDP growth estimate evolved week-by-week
        # as new economic data arrived during the quarter.
        if not nowcast_df.empty:
            fig = px.line(
                nowcast_df,
                x="forecast_date",
                y="nowcast",
                title=f"NY Fed GDP Nowcast: {yr} {qtr}",
                labels={
                    "forecast_date": "Forecast Date", "nowcast": "Annualised GDP Growth (%)"
                    }
            )
            fig.update_traces(
                line=dict(width=3, color="#1f77b4"),
                name = "NY Fed Nowcast",
                showlegend=True,
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Nowcast:</b> %{y:.2f}%<extra></extra>",
            )
        else:
            # If no data, create an empty figure with just axes and title
            fig = px.line(
                title=f"NY Fed GDP Nowcast: {yr} {qtr}"
            )
        # ---- Optional comparison overlays ----
        # Only drawn when the "Compare Data" checkbox is checked and
        # at least one comparison series is selected.
        if input.compare_data() and input.comparison_series() and not horizon_df.empty:
            overlay_config = {
                # key : (legend label, line dash style, colour, column name)
                "backcast": ("Backcast (prev quarter)", "dash",  "#2ca02c", "backcast"),
                "forecast": ("Forecast (next quarter)", "dot",   "#ff7f0e", "forecast"),
            }
            for key in input.comparison_series():
                if key in overlay_config:
                    label, dash_style, color, col = overlay_config[key]
                    series = horizon_df[["forecast_date", col]].dropna(subset=[col])
                    if not series.empty:
                        fig.add_scatter(
                           
                            x=series["forecast_date"],
                            y=series[col],
                            mode="lines",
                            name=label,
                            line=dict(width=2, dash=dash_style, color=color),
                            hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>{label}:</b> %{{y:.2f}}%<extra></extra>",

                        )

        # ---- Layout settings ----
        fig.update_layout(
            xaxis=dict(
                # type = "date" #To fix UNIX date formatting issues, but it causes dtick/tickformat to break, so we format dates in hovertemplate instead
                title="Forecast Date",
                # dtick="M1",  # tick every month
                # tickformat="%b %Y",
                tickangle=-45,
                autorange=True
            ),
            yaxis=dict(
                title="Annualised GDP Growth (%)",
                zeroline=True,
                zerolinecolor="#999999",
                zerolinewidth=1,
                autorange=True
            ),
            legend=dict(
                orientation="h",
                x=0,
                y=-0.2,
                font=dict(size=12)
            ),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig


# =============================================================================
# CREATE AND RUN THE APP
#
# App() ties the UI and server together — equivalent of R's shinyApp(ui, server).
# Run with:  shiny run app.py
# Then open http://127.0.0.1:8000 in your browser.
# =============================================================================

app = App(app_ui, server)
