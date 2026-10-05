//! Finance tools (Python `src/tools/finance.py` port): yfinance-style data
//! download (via `yahoo_finance_api`) and chart plotting (`plotters`).

use crate::log_call;
use crate::tools::files::safe_filename;
use crate::tools::registry::{ToolDef, ToolOutput};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

/// Yahoo Finance time period definitions (Python `ValidYF.period`).
pub const PERIODS: &[&str] = &[
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
];
/// Yahoo Finance time interval definitions (Python `ValidYF.interval`).
pub const INTERVALS: &[&str] = &[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h", "1d", "5d", "1wk", "1mo", "3mo",
];

/// Fetch closing price of a stock symbol within a time period with a time
/// interval. Useful when user asks for getting financial info of a financial
/// product given its symbol.
///
/// Returns the `Date | Open | Close` rows rendered like pandas `.to_string()`
/// (or an error message).
pub async fn fetch_financial_data(
    symbol: &str,
    interval: &str,
    period: &str,
    start_date: Option<&str>,
    end_date: Option<&str>,
    to_string: bool,
) -> Result<String> {
    log_call!("fetch_financial_data", "symbol" => symbol, "interval" => interval, "period" => period, "start_date" => start_date.unwrap_or_default(), "end_date" => end_date.unwrap_or_default());
    if !INTERVALS.contains(&interval) {
        return Ok(format!("Error: invalid interval '{interval}'"));
    }
    if !PERIODS.contains(&period) && start_date.is_none() && end_date.is_none() {
        return Ok(format!("Error: invalid period '{period}'"));
    }
    let connector =
        yahoo_finance_api::YahooConnector::new().map_err(|e| anyhow::anyhow!("Error: {e}"))?;
    let response = match (start_date, end_date) {
        (Some(start), Some(end)) => {
            let start = parse_date(start)?;
            let end = parse_date(end)?;
            connector
                .get_quote_history_interval(symbol, start, end, interval)
                .await
        }
        _ => {
            connector
                .get_quote_period_interval(symbol, period, interval, false)
                .await
        }
    };
    let response = match response {
        Ok(r) => r,
        Err(e) => return Ok(format!("Error: {e}")),
    };
    let quotes = match response.quotes() {
        Ok(q) => q,
        Err(e) => return Ok(format!("Error: {e}")),
    };
    if quotes.is_empty() {
        return Ok(format!("No data found for {symbol}."));
    }
    let rows: Vec<String> = quotes
        .iter()
        .filter_map(|q| {
            let date = chrono::DateTime::from_timestamp(q.timestamp, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())?;
            Some(format!("{date}  {:.2}  {:.2}", q.open, q.close))
        })
        .collect();
    if rows.is_empty() {
        return Ok("Not all the required columns ['Date', 'Open', 'Close'] are available.".into());
    }
    if !to_string {
        // The Python tool returns a DataFrame when to_string=False; agent tools
        // always use the string form (parity of the exposed tool behavior).
    }
    Ok(rows.join("\n"))
}

/// Parse a `YYYY-MM-DD` date into the `time::OffsetDateTime` the connector
/// expects (yahoo_finance_api re-exports the `time` crate).
fn parse_date(date: &str) -> Result<yahoo_finance_api::time::OffsetDateTime, anyhow::Error> {
    let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("Error: bad date '{date}': {e}"))?
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("Error: bad date '{date}'"))?;
    let unix = date.and_utc().timestamp();
    yahoo_finance_api::time::OffsetDateTime::from_unix_timestamp(unix)
        .map_err(|e| anyhow::anyhow!("Error: bad date '{date}': {e}"))
}

/// Fetch closing price of a stock symbol within a time period with a time
/// interval. The closing price are plotted in a chart that is saved as image.
/// Useful when user asks for plotting a financial product given its symbol.
pub async fn save_financial_chart(
    symbol: &str,
    interval: &str,
    period: &str,
    out_dir: Option<&str>,
) -> String {
    log_call!("save_financial_chart", "symbol" => symbol, "interval" => interval, "period" => period, "out_dir" => out_dir.unwrap_or_default());
    let rows = fetch_chart_rows(symbol, interval, period).await;
    let rows = match rows {
        Err(e) => return format!("Error: {e}"),
        Ok(r) => r,
    };
    let out_directory: PathBuf = match out_dir {
        Some(dir) => PathBuf::from(dir),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };
    let _ = std::fs::create_dir_all(&out_directory);
    let plot_path = out_directory.join(format!("{}.jpg", safe_filename(symbol)));
    match render_chart(&rows, symbol, &plot_path) {
        Err(e) => format!("Error: {e}"),
        Ok(_) => format!("Plot saved to '{}'", plot_path.display()),
    }
}

type ChartRows = Vec<(chrono::NaiveDateTime, f64)>;

async fn fetch_chart_rows(symbol: &str, interval: &str, period: &str) -> anyhow::Result<ChartRows> {
    let connector = yahoo_finance_api::YahooConnector::new()?;
    let response = connector
        .get_quote_period_interval(symbol, period, interval, false)
        .await?;
    let quotes = response.quotes()?;
    let mut rows = Vec::new();
    for q in quotes {
        if let Some(dt) = chrono::DateTime::from_timestamp(q.timestamp, 0) {
            rows.push((dt.naive_utc(), q.close));
        }
    }
    if rows.is_empty() {
        return Err(anyhow::anyhow!("No data found for {symbol}."));
    }
    Ok(rows)
}

/// Draw the close-price line chart and save it as JPG (plotters).
/// The x axis is unix seconds (plotters needs a `Ranged` type; f64 keeps the
/// dependency surface small while the date formatting stays readable).
fn render_chart(rows: &ChartRows, symbol: &str, path: &PathBuf) -> anyhow::Result<()> {
    use plotters::prelude::*;
    let as_secs = |dt: &chrono::NaiveDateTime| dt.and_utc().timestamp() as f64;
    let first = as_secs(&rows.first().unwrap().0);
    let last = as_secs(&rows.last().unwrap().0);
    let min_close = rows.iter().map(|r| r.1).fold(f64::MAX, f64::min);
    let max_close = rows.iter().map(|r| r.1).fold(f64::MIN, f64::max);
    let root = BitMapBackend::new(path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{symbol} Price Over Time"),
            ("sans-serif", 24).into_font(),
        )
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(first..last, min_close..max_close)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(8)
        .x_label_formatter(&|secs| {
            chrono::DateTime::from_timestamp(*secs as i64, 0)
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_default()
        })
        .draw()?;
    chart.draw_series(LineSeries::new(
        rows.iter().map(|(d, c)| (as_secs(d), *c)),
        &BLUE,
    ))?;
    chart.draw_series(
        rows.iter()
            .map(|(d, c)| Circle::new((as_secs(d), *c), 3, BLUE.mix(0.8))),
    )?;
    root.present()?;
    Ok(())
}

// ------------------------------------------------------------- registry

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "fetch_financial_data".into(),
            description: "Fetch closing price of a stock symbol within a time period with a time interval.\nUseful when user asks for getting financial info of a financial product given its symbol.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "default": "1mo"},
                    "period": {"type": "string", "default": "1y"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "to_string": {"type": "boolean", "default": true}
                },
                "required": ["symbol"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let symbol = args.get("symbol").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let interval = args.get("interval").and_then(|v| v.as_str()).unwrap_or("1mo").to_string();
                let period = args.get("period").and_then(|v| v.as_str()).unwrap_or("1y").to_string();
                let start = args.get("start_date").and_then(|v| v.as_str()).map(str::to_string);
                let end = args.get("end_date").and_then(|v| v.as_str()).map(str::to_string);
                let text = fetch_financial_data(&symbol, &interval, &period, start.as_deref(), end.as_deref(), true)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}"));
                Ok(ToolOutput::text(text))
            })),
        },
        ToolDef {
            name: "save_financial_chart".into(),
            description: "Fetch closing price of a stock symbol within a time period with a time interval.\nThe closing price are plotted in a chart that is saved as image.\nUseful when user asks for plotting a financial product given its symbol.".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "default": "1mo"},
                    "period": {"type": "string", "default": "1y"},
                    "out_dir": {"type": "string"}
                },
                "required": ["symbol"]
            }),
            handler: Arc::new(|_ctx, args| Box::pin(async move {
                let symbol = args.get("symbol").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let interval = args.get("interval").and_then(|v| v.as_str()).unwrap_or("1mo").to_string();
                let period = args.get("period").and_then(|v| v.as_str()).unwrap_or("1y").to_string();
                let out_dir = args.get("out_dir").and_then(|v| v.as_str()).map(str::to_string);
                Ok(ToolOutput::text(save_financial_chart(&symbol, &interval, &period, out_dir.as_deref()).await))
            })),
        },
    ]
}
