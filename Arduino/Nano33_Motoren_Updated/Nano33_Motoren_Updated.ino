// Nano 33 BLE servo control with separate calibration per servo.
// Same commands as the original sketch: 'f' forward, 'b' backward, 's' stop.
// Serial: 9600 baud.

#include <Servo.h>

Servo servoRight;
Servo servoLeft;

byte pinFeedbackLeft = 2;
byte pinServoLeft = 3;
byte pinFeedbackRight = 4;
byte pinServoRight = 5;

// ---- Calibration ----
int stopLeft  = 94;   // left wheel stands still
int stopRight = 93;   // right wheel stands still

// Forward (already almost perfect)
int speedLeftFwd   = 6;
int speedRightFwd  = 9;

// Backward (tune separately)
int speedLeftBack  = 12;   // left was too slow -> increase
int speedRightBack = 7;   // or decrease this if right is still faster

unsigned long spinTime = 1010;   // ms for one full 360° turn -> tune this

// --------------------------------------------------------------------

// Servos are mounted mirrored, so "forward" is + on the left and - on the right
int forwardsLeft   = stopLeft  + speedLeftFwd;
int backwardsLeft  = stopLeft  - speedLeftBack;
int forwardsRight  = stopRight - speedRightFwd;
int backwardsRight = stopRight + speedRightBack;

char keyCommand;

void drive(int valueLeft, int valueRight) {
  servoRight.attach(pinServoRight);
  servoLeft.attach(pinServoLeft);
  servoRight.write(valueRight);
  servoLeft.write(valueLeft);
  delay(500);
  servoRight.detach();
  servoLeft.detach();
}

void setup() {
  Serial.begin(9600);
  pinMode(pinFeedbackRight, INPUT);
  pinMode(pinFeedbackLeft, INPUT);
}

void loop() {
  if (Serial.available() > 0) {
    keyCommand = Serial.read();

    switch (keyCommand) {
      case 'f': drive(forwardsLeft,  forwardsRight);  break;
      case 'b': drive(backwardsLeft, backwardsRight); break;
      case 's': drive(stopLeft,      stopRight);      break;
      case 'g': // spin 360° on the spot: left forward, right backward
                servoRight.attach(pinServoRight);
                servoLeft.attach(pinServoLeft);
                servoLeft.write(forwardsLeft);
                servoRight.write(backwardsRight);
                delay(spinTime);
                servoRight.detach();
                servoLeft.detach();
                break;
      default: break;   // ignores '\n', '\r' and unknown characters
    }
  }
}
