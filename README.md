# Photogenix Retouch

# Test locally
docker build -t 'retouch_api:latest' .
docker run -d -p 5007:5007 retouch_api

# Set current project
gcloud config set project nomadic-basis-452012-v8

# Create repo once, currently not needed for this repo as it's already created
gcloud artifacts repositories create generative-service --repository-format=docker \
  --location=asia-south1
  
# Authenticate docker with GCP
gcloud auth configure-docker asia-south1-docker.pkg.dev

# List repositories under current project
gcloud artifacts repositories list

# Tag your image
docker build -t asia-south1-docker.pkg.dev/nomadic-basis-452012-v8/generative-service/retouch-api .

# Push image
docker push asia-south1-docker.pkg.dev/nomadic-basis-452012-v8/generative-service/retouch-api
# format -  <region/projet-id/repo-id/service-name>

# Deploy service
gcloud run deploy retouch-api \
  --image=asia-south1-docker.pkg.dev/nomadic-basis-452012-v8/generative-service/retouch-api \
  --platform=managed \
  --region=asia-south1 \
  --allow-unauthenticated \
  --port=5007 \
  --min-instances 1 \
  --max-instances 100 \
  --concurrency 1 \
  --timeout 300 \
  --cpu 2 \
  --memory 4Gi \
  --execution-environment gen2 \
  --no-cpu-throttling \
  --cpu-boost \
  --set-env-vars=GEMINI_KEY=$GEMINI_FLASH

# Notes
* Masked inpaint api - https://retouch-api-132358776415.asia-south1.run.app/retouch/masked/generate
* Fastapi docs - https://retouch-api-132358776415.asia-south1.run.app/retouch/docs
* Ensure gemini key is available in env var GEMINI_FLASH at cloudrun deploy stage
