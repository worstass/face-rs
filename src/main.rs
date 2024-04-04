use axum::{response::Html, routing::get, Router};
use axum::{
    error_handling::HandleErrorLayer,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, any, post, patch},
    Json, Router,
};
use base64::prelude::*;
use std::net::SocketAddr;
use tokio::signal;
use serde::{Deserialize, Serialize};


#[tokio::main]
async fn main() {
    let app = Router::new()
        .fallback(handler_404)
        .route("/healthcheck", any(healthcheck))
        .route("/status", get(status))
        .route("/find_faces_base64", post(find_faces_base64))
        .route("/find_faces", post(find_faces))
        .route("/scan_faces", post(scan_faces))
        .route("/", get(show_form).post(accept_form));

    // run it
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn healthcheck() -> Json<HealthCheck> {
    Json(HealthCheck {
        status: "ok".to_string()
    })
}

async fn status() -> Json<Status> {
    Json(Status {
        status: "ok".to_string(),
        build_version: todo!(),
        calculator_version: todo!(),
        similarity_coefficients: todo!(),
        available_plugins: todo!(),
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("signal received, starting graceful shutdown");
}

async fn handler_404() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, "nothing to see here")
}

#[derive(Debug, Deserialize)]
struct FaceResult {
    plugins_versions: String,
    calculator_version: String,
    result: String,
}

#[derive(Debug, Deserialize)]
struct FaceInput {
    file: String,
    detect_faces: bool,
    limit: u16, //LIMIT = 'limit'
    det_prob_threshold: f16, //DET_PROB_THRESHOLD = 'det_prob_threshold'
    face_plugins: String, //FACE_PLUGINS = 'face_plugins'
}

#[derive(Debug, Deserialize)]
struct HealthCheck {
    status: String,
}

#[derive(Debug, Deserialize)]
struct Status {
    status: String,
    build_version: String, //ENV.BUILD_VERSION,
    calculator_version: String, // //=str(calculator),
    similarity_coefficients: String, //=calculator.ml_model.similarity_coefficients,
    available_plugins: String, //=available_plugins
}

async fn scan_faces(
    pagination: Option<Query<Pagination>>,
    // State(db): State<Db>,
) -> impl IntoResponse {
    let result = FaceResult { plugins_versions: todo!(), calculator_version: todo!(), result: todo!() };
    Json(result)
}

async fn find_faces_base64(
    Json(input): Json<FaceInput>,
    // pagination: Option<Query<Pagination>>,
    // State(db): State<Db>,
) -> impl IntoResponse {
    let raw = BASE64_STANDARD.decode(input.file.as_bytes());

    let result = FaceResult { plugins_versions: todo!(), calculator_version: todo!(), result: todo!() };
    Json(result)
}

async fn find_faces(mut multipart: Multipart)  -> impl IntoResponse {
    while let Some(mut field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string(); // == file
        let data = field.bytes().await.unwrap();

        facerecog::run();

        println!("Length of `{}` is {} bytes", name, data.len());
        let result = FaceResult { plugins_versions: todo!(), calculator_version: todo!(), result: todo!() };
        Json(result)
    }
}