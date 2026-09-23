# syntax=docker/dockerfile:1.7

FROM rust:1.92-bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --locked --release -p huginn-server

FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="Huginn" \
      org.opencontainers.image.description="Copenhagen and five-dimensional hnefatafl server" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 huginn \
    && useradd --uid 10001 --gid huginn --no-create-home --shell /usr/sbin/nologin huginn \
    && install --directory --owner huginn --group huginn /data

COPY --from=builder /src/target/release/huginn-server /usr/local/bin/huginn-server

USER huginn:huginn
WORKDIR /data

ENV HUGINN_ADDR=0.0.0.0:3000 \
    HUGINN_DATABASE=/data/huginn.sqlite3 \
    RUST_LOG=info

VOLUME ["/data"]
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:3000/api/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/huginn-server"]
