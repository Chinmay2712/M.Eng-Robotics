#!/bin/bash
# camera_setup.sh - fixed camera settings for the Logitech C505
# Run it after every plug-in / reboot (camera node may run already):
#     ./camera_setup.sh                    lab values, /dev/video0
#     ./camera_setup.sh dorm               dorm values (darker room)
#     ./camera_setup.sh lab /dev/video2    other device
#     ./camera_setup.sh 120                any exposure (units of 100 us, 120 = 12 ms)
# The settings are lost when the camera is unplugged.
# Exposure above about 300 (30 ms) lowers the frame rate below 30 fps and blurs when the robot moves.

MODE="${1:-lab}"
DEV="${2:-/dev/video0}"

EXPOSURE_LAB=150      # lab, 2026-11-23 (lane detection worked well)
EXPOSURE_DORM=105     # dorm, 2026-11-21
GAIN=64

case "$MODE" in
  lab)  EXPOSURE=$EXPOSURE_LAB ;;
  dorm) EXPOSURE=$EXPOSURE_DORM ;;
  *[!0-9]*|'') echo "Unknown mode '$MODE' (use lab, dorm or a number)"; exit 1 ;;
  *)    EXPOSURE=$MODE ;;
esac

if [ ! -e "$DEV" ]; then
  echo "Camera device $DEV not found (ls /dev/video*, v4l2-ctl --list-devices)"
  exit 1
fi

set_ctrl() {
  if ! v4l2-ctl -d "$DEV" --set-ctrl="$1"; then
    echo "WARNING: could not set $1"
  fi
}

# order matters: manual exposure mode first, then the exposure time
set_ctrl auto_exposure=1                 # 1 = manual mode
set_ctrl exposure_dynamic_framerate=0    # keep the frame rate constant in dim light
set_ctrl exposure_time_absolute=$EXPOSURE   # units of 100 us
set_ctrl gain=$GAIN
set_ctrl power_line_frequency=1          # 50 Hz mains (Germany): no flicker bands
set_ctrl white_balance_automatic=0
set_ctrl white_balance_temperature=4000

# ask for 30 frames per second
v4l2-ctl -d "$DEV" --set-parm=30

echo
echo "Camera $DEV configured: exposure $EXPOSURE, gain $GAIN. Check with:  v4l2-ctl -d $DEV --list-ctrls"
