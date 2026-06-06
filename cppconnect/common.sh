export PROJECT=${PROJECT:-"Colosseum"}
export LOGGING_PROJECT=${PROJECT:-"$PROJECT"}
#export CLUSTER_NAME=${CLUSTER_NAME:-"minigo-v5"}
#export ZONE=${ZONE:-"asia-east1-a"}
#export NUM_K8S_NODES=${NUM_NODES:-"5"}

# Configuration for service accounts so that the cluster can do cloud-things.
#export SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"${PROJECT}-${CLUSTER_NAME}-services"}
#export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"
#export SERVICE_ACCOUNT_KEY_LOCATION=${SERVICE_ACCOUNT_KEY_LOCATION:-"/tmp/${SERVICE_ACCOUNT}-key.json"}

# Constants for docker container creation
#export VERSION_TAG=${VERSION_TAG:-"0.16"}
#export EVAL_VERSION_TAG=${EVAL_VERSION_TAG:-"latest"}
#export MINIGUI_PY_CPU_CONTAINER=${MINIGUI_PY_CPU_CONTAINER:-"minigui-py-cpu-v2"}

# Bucket names live in a single global namespace
# So, we prefix the project name to avoid collisions
#
# For more details, see https://cloud.google.com/storage/docs/best-practices
export BUCKET_NAME=${BUCKET_NAME:-"${PROJECT}-colosseum"}

# By default, buckets are created in us-east1, but for more performance, it's
# useful to have a region located near the GKE cluster.
# For more about locations, see
# https://cloud.google.com/storage/docs/bucket-locations
#export BUCKET_LOCATION=${BUCKET_LOCATION:-"asia-east1"}