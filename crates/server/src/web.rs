use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

const INDEX: &str = include_str!("../../../web/index.html");
const APP: &str = include_str!("../../../web/app.js");
const STYLES: &str = include_str!("../../../web/styles.css");
const FAVICON: &str = include_str!("../../../web/favicon.svg");

pub(crate) async fn index() -> Response {
    asset("text/html; charset=utf-8", INDEX)
}

pub(crate) async fn app() -> Response {
    asset("text/javascript; charset=utf-8", APP)
}

pub(crate) async fn styles() -> Response {
    asset("text/css; charset=utf-8", STYLES)
}

pub(crate) async fn favicon() -> Response {
    asset("image/svg+xml", FAVICON)
}

pub(crate) async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, "Not found").into_response()
}

fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'",
            ),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_client_contains_coordinate_grid_and_all_board_renderer() {
        assert!(INDEX.contains("id=\"multiverse\""));
        assert!(APP.contains("timeline.row - minRow + 2"));
        assert!(APP.contains("coordinate.time - minTime + 2"));
        assert!(APP.contains("Object.values(timeline.boards)"));
        assert!(STYLES.contains("repeat(var(--timeline-count)"));
        assert!(FAVICON.starts_with("<svg"));
    }
}
