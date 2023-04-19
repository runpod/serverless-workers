FROM merrell/ait-sd-1-runpod:anything as anything
FROM merrell/ait-sd-1-runpod:latest
COPY --from=anything /tmp /tmp