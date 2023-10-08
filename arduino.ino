
#include <Servo.h>
#include <AccelStepper.h>
#include <SoftwareSerial.h>


Servo servo;
SoftwareSerial blue(2,3);
int initialPosition = 0;
int angle = 0;
const int servoPin = 9;
const int sensorPin = 13;
int dataQueue[100];  
int queueSize = 0;  
int Step = 5; int Dir = 4; int ena = 7;
const int sensorPin1 = 12;
AccelStepper mystepper(1, Step, Dir, ena);

void setup() {
  Serial.begin(9600);
  blue.begin(9600);

  servo.attach(servoPin);
  servo.write(initialPosition);
  pinMode(sensorPin, INPUT);
  mystepper.setMaxSpeed(1000);
  pinMode(sensorPin1, INPUT);


}

void loop() {
//  Serial.println(digitalRead(2));
  if (blue.available()) {
    int signal = blue.read();
    dataQueue[queueSize] = signal;  
    queueSize++;
  }
  if (digitalRead(sensorPin) == 0 && queueSize > 0) {
    int signal = dataQueue[0];  
    
    if (signal == '1') {
      angle = 0;
    } else if (signal == '2') {
      angle = 40;
    } else if (signal == '3') {
      angle = 70;
    }
    
    servo.write(initialPosition + angle);
    delay(500);
    
    
    for (int i = 0; i < queueSize - 1; i++) {
      dataQueue[i] = dataQueue[i + 1];
    }
    queueSize--;
  }

  if (digitalRead(sensorPin1) == 0){
  mystepper.setCurrentPosition(0); //Set the current position of the motor to 0
  while(mystepper.currentPosition() != -40) // 400 bước = 2 vòng ..... currentPosition() return the current position of the motor
  {
    mystepper.setSpeed(-90); 
    mystepper.runSpeed();
  }
  
  delay(2200); 
   }
}
