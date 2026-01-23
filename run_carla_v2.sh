#!/usr/bin/env bash

# --- Configuration ---
# Default values for the script parameters.
DEFAULT_GPU_ID=0
DEFAULT_USE_DISPLAY=false
# ---------------------

# Function to display usage information
print_usage() {
  echo "Usage: $(basename "$0") -p <port> -v <version> [-d] [-g <gpu_id>] [-h]"
  echo
  echo "Starts a CARLA simulator in a Docker container."
  echo
  echo "Required arguments:"
  echo "  -p <port>      The RPC port for the CARLA server."
  echo "  -v <version>   The CARLA simulator version tag (e.g., 0.9.13, 0.9.14)."
  echo
  echo "Optional arguments:"
  echo "  -d             Enable CARLA display (GUI mode). Defaults to off-screen rendering."
  echo "  -g <gpu_id>    The GPU device ID to use. Defaults to ${DEFAULT_GPU_ID}."
  echo "  -h             Display this help message and exit."
  echo
  echo "Example:"
  echo "  # Run CARLA 0.9.13 off-screen on port 2000 using GPU 0"
  echo "  $(basename "$0") -p 2000 -v 0.9.13"
  echo
  echo "  # Run CARLA 0.9.14 with display enabled on port 2005 using GPU 1"
  echo "  $(basename "$0") -p 2005 -v 0.9.14 -d -g 1"
}

main() {
  # --- Set initial variables from defaults ---
  local port=""
  local carla_version=""
  local use_display=${DEFAULT_USE_DISPLAY}
  # Use CUDA_VISIBLE_DEVICES if set, otherwise use the default GPU ID
  local gpu_id=${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPU_ID}}

  # --- Parse Command-Line Arguments ---
  while getopts ":p:v:g:dh" opt; do
    case ${opt} in
      p) port=$OPTARG ;;
      v) carla_version=$OPTARG ;;
      g) gpu_id=$OPTARG ;;
      d) use_display=true ;;
      h) print_usage; exit 0 ;;
      \?) echo "❌ Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
      :) echo "❌ Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
    esac
  done

  # --- Validate Required Arguments ---
  # Also check CARLA_PORT_DOCKER for backward compatibility if -p is not set
  port=${port:-$CARLA_PORT_DOCKER}

  if [ -z "$port" ]; then
    echo "❌ Error: Port is a required argument. Please provide it with -p or set CARLA_PORT_DOCKER." >&2
    print_usage
    exit 1
  fi

  if [ -z "$carla_version" ]; then
    echo "❌ Error: CARLA version is a required argument. Please provide it with -v." >&2
    print_usage
    exit 1
  fi

  # --- Prepare Docker Command ---
  local container_name="carla-sim-${port}"
  local carla_image="carlasim/carla:${carla_version}"

  # Base Docker options
  local docker_opts=(
    -d
    --name="$container_name"
    --gpus "device=${gpu_id}"
    --net=host
  )
  
  # Base CARLA command
  local carla_cmd=(
    ./CarlaUE4.sh
    -carla-rpc-port="$port"
    -quality-level=Epic
  )

  echo "--- CARLA Configuration ---"
  echo "➡️  Port:           ${port}"
  echo "➡️  Version:        ${carla_version}"
  echo "➡️  Container Name: ${container_name}"
  echo "➡️  GPU ID:         ${gpu_id}"

  # --- Configure for Display or Off-Screen Rendering ---
  if [ "$use_display" = true ]; then
    if [ -z "$DISPLAY" ]; then
      echo "⚠️  Warning: -d flag was used, but the DISPLAY environment variable is not set."
      echo "           Defaulting to off-screen rendering."
      use_display=false
    else
      echo "➡️  Display Mode:   Enabled (GUI)"
      docker_opts+=(-e DISPLAY="$DISPLAY" --privileged)
      carla_cmd+=(-ResX=800 -ResY=600)
    fi
  fi

  if [ "$use_display" = false ]; then
    echo "➡️  Display Mode:   Disabled (Off-Screen)"
    carla_cmd+=(-RenderOffScreen)
  fi
  echo "--------------------------"
  echo

  # --- Execute Docker Commands ---
  echo "🔄 Stopping and removing any existing container named '${container_name}'..."
  docker rm -f "$container_name" 2>/dev/null || true # Suppress error if container doesn't exist

  echo "🚀 Launching new CARLA container..."
  echo
  
  # The final command is printed for debugging and clarity
  set -x 
  docker run "${docker_opts[@]}" "$carla_image" "${carla_cmd[@]}"
  set +x
  
  echo
  echo "✅ CARLA container '${container_name}' has been started."
  echo "   To view logs, run: docker logs -f ${container_name}"
}

# Run the main function
main "$@"