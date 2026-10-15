//! Bounded, engine-computed Insights charts shared with graphical clients.

// Rust guideline compliant 2026-02-21
use serde::{Deserialize, Serialize};

/// The statistic and units represented by one chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Tool invocation counts.
    Tools,
    /// File touches by language.
    Languages,
    /// Sessions by project.
    Projects,
    /// Failures grouped by category.
    Errors,
    /// Failures grouped by tool.
    ToolFailures,
    /// Distribution of response times.
    ReplyTimes,
    /// Failed invocations as percentages.
    FailureRates,
    /// Human messages by host-local hour.
    PromptsByHour,
    /// Positive ratings by host-local date.
    Satisfaction,
}

impl Kind {
    /// Returns a user-facing chart title.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Tools => "Tools you use",
            Self::Languages => "Languages",
            Self::Projects => "Projects",
            Self::Errors => "Sources of friction",
            Self::ToolFailures => "Tool failures",
            Self::ReplyTimes => "Reply times",
            Self::FailureRates => "Tool failure rates",
            Self::PromptsByHour => "Prompts by host-local hour",
            Self::Satisfaction => "Positive ratings by host-local day",
        }
    }

    /// Returns whether measurements use a fixed 0–100 percent scale.
    #[must_use]
    pub fn percent(self) -> bool {
        matches!(self, Self::FailureRates | Self::Satisfaction)
    }
}

/// One bounded chart whose data comes from deterministic session statistics.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Chart {
    /// Statistic and units.
    pub kind: Kind,
    /// Display labels and numeric values in engine order.
    pub rows: Vec<(String, f64)>,
    /// Coverage caveat, or an empty string.
    pub note: String,
}

impl std::fmt::Debug for Chart {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Chart")
            .field("kind", &self.kind)
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}

impl Chart {
    /// Validates finite measurements, units, and bounded display text.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.rows.is_empty()
            && self.rows.len() <= 32
            && self.note.len() <= 512
            && !self.note.chars().any(char::is_control)
            && self.rows.iter().all(|(label, value)| {
                !label.trim().is_empty()
                    && label.len() <= 256
                    && !label.chars().any(char::is_control)
                    && value.is_finite()
                    && *value >= 0.0
                    && if self.kind.percent() {
                        *value <= 100.0
                    } else {
                        *value <= f64::from(u32::MAX) && value.fract().abs() <= f64::EPSILON
                    }
            })
    }

    /// Returns a nonzero axis maximum, preserving fixed percentage scales.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.kind.percent() {
            100.0
        } else {
            self.rows
                .iter()
                .map(|(_, value)| *value)
                .filter(|value| value.is_finite())
                .fold(1.0_f64, f64::max)
        }
    }
}

/// Validates the complete chart inventory, including unique statistic kinds.
#[must_use]
pub fn valid_charts(charts: &[Chart]) -> bool {
    let mut kinds = std::collections::HashSet::new();
    charts.len() <= 9
        && charts
            .iter()
            .all(|chart| chart.is_valid() && kinds.insert(chart.kind))
}

#[cfg(test)]
mod tests {
    use super::{Chart, Kind, valid_charts};

    #[test]
    fn insights_charts_validate_units_bounds_identity_and_safe_scales() {
        let mut chart = Chart {
            kind: Kind::Tools,
            rows: vec![("read".into(), 0.0)],
            note: String::new(),
        };
        assert!(chart.is_valid());
        assert!((chart.scale() - 1.0).abs() < f64::EPSILON);
        for value in [
            f64::NAN,
            f64::INFINITY,
            -1.0,
            0.5,
            f64::from(u32::MAX) + 1.0,
        ] {
            chart.rows[0].1 = value;
            assert!(!chart.is_valid());
        }
        chart.kind = Kind::FailureRates;
        chart.rows[0].1 = 0.5;
        assert!(chart.is_valid());
        assert!((chart.scale() - 100.0).abs() < f64::EPSILON);
        chart.rows[0].1 = 100.1;
        assert!(!chart.is_valid());
        chart.rows[0].1 = 100.0;
        assert!(!valid_charts(&[chart.clone(), chart.clone()]));
        assert!(valid_charts(std::slice::from_ref(&chart)));
        chart.rows[0].0 = "x".repeat(257);
        assert!(!chart.is_valid());
        chart.rows[0].0 = "bad\nlabel".into();
        assert!(!chart.is_valid());
        chart.rows = vec![("read".into(), 1.0); 33];
        assert!(!chart.is_valid());
    }

    #[test]
    fn insights_charts_roundtrip_and_debug_omit_project_labels_and_notes() {
        let chart = Chart {
            kind: Kind::Projects,
            rows: vec![("private-project".into(), 17.0)],
            note: "private-note".into(),
        };
        let debug = format!("{chart:?}");
        assert!(!debug.contains("private-project"));
        assert!(!debug.contains("private-note"));
        let value = serde_json::to_value(&chart).unwrap();
        assert_eq!(serde_json::from_value::<Chart>(value).unwrap(), chart);
    }
}
