
#include <Wire.h>
#include "MPU9250.h"

#define I2Cclock 400000
#define I2Cport Wire
#define MPU9250_ADDRESS MPU9250_ADDRESS_AD0  // 0x68
 
MPU9250 myIMU(MPU9250_ADDRESS, I2Cport, I2Cclock);

// مصفوفة عشان ناخد معايرة الماجنتوميتر
float magCalibration[3] = { 0, 0, 0 };

// ---- Yaw بالجايرو ----  = z angle = head angle of magno
float yaw = 0;
unsigned long lastTime = 0;
float gyroBiasZ = 0;

// Ultrasonic Pins
#define trigFront 48
#define echoFront 49
#define trigRight 50
#define echoRight 51
#define trigLeft 52
#define echoLeft 53

// Motor Pins
#define MFL1 28
#define MFL2 29
#define MFR1 26
#define MFR2 27
#define MBL1 24
#define MBL2 25
#define MBR1 22
#define MBR2 23
//motor enable pins (speed pwm)
#define MFRSP 5
#define MFLSP 4
#define MBRSP 2
#define MBLSP 3

#define sensorIn A2   //digital input from the metal sensor
#define sensorOut A3  //digital output to the buzzer circuit

// last pulse time for debounce (microseconds)
volatile unsigned long lastLeftMicros = 0;
volatile unsigned long lastRightMicros = 0;

// minimal valid interval between pulses (microseconds)
const unsigned long MIN_PULSE_US = 1000UL;  // 1 ms

//encoder settings
//const int pulsesPerRev = 20;   // الإنكودر بتاعك 20 نبضة
//float wheelDiameter = 12;     // قطر العجلة (سم)
//distanceStep = 2*pi*r/20
#define distanceStep 0.0188495  //m
#define CARRadius 0.27          //m
volatile int pulsesL = 0;
volatile int pulsesR = 0;

// ========== متغيرات الزجزاج ==========
float sweepHeight = 2;  // الطول (م)
float sweepWidth = 2;   // العرض (م)
float carWidth = 0.445;    // عرض الروبوت (م)
float lineWidth = carWidth;

int linesNum = sweepWidth / lineWidth;
int currentHLine = 0;
int speed = 150;
int sp_rotate=150;
float xPosition = 0;
float yPosition = 0;
float supposedAngle = 0;
float currentAngle = 0;
float distance = 0;
unsigned long yawAngleStart = 0;
unsigned long avgHeadStart = 0;
unsigned long headStart = 0;
float heading = 0;
float sum = 0;
int counter = 0;
bool detectStart = false;
bool isAuto = false;
unsigned long manualStart;         //millis
unsigned long detectionStartTime;  //millis
bool objectDetected = false;
int mLDirection = 1;
int mRDirection = 1;
String data ="";

enum States { straight, turning, finished , paused, bypassTarget};
States state = straight;
States lastState;

enum Axis { X, Y};
struct Orientation 
{
 Axis axis;
 int sign;
};

Orientation curOrient = { X, 1};
Orientation prevOrient = { Y, 1};

int rotSign = 1;

float yawOffset = 0; //for manual Yaw (currentAngle) reset

// ====== Setup ======
void setup() 
{
  Serial.begin(9600);
  Serial1.begin(9600);

  Wire.begin();

  pinMode(trigFront, OUTPUT);
  pinMode(echoFront, INPUT);
  pinMode(trigRight, OUTPUT);
  pinMode(echoRight, INPUT);
  pinMode(trigLeft, OUTPUT);
  pinMode(echoLeft, INPUT);

  pinMode(MFL1, OUTPUT);
  pinMode(MFL2, OUTPUT);
  pinMode(MFR1, OUTPUT);
  pinMode(MFR2, OUTPUT);
  pinMode(MBL1, OUTPUT);
  pinMode(MBL2, OUTPUT);
  pinMode(MBR1, OUTPUT);
  pinMode(MBR2, OUTPUT);

  pinMode(MFRSP, OUTPUT);
  pinMode(MFLSP, OUTPUT);
  pinMode(MBRSP, OUTPUT);
  pinMode(MBLSP, OUTPUT);

  pinMode(sensorIn, INPUT);
  pinMode(sensorOut, OUTPUT);

  analogWrite(MFRSP, speed);
  analogWrite(MFLSP, speed);
  analogWrite(MBRSP, speed);
  analogWrite(MBLSP, speed);

  //pin A8 = left encoder,,, pin A9 = Right encoder
  PCICR |= B00000100;   //enable group1 pin changed interrupts from PCINT8-14
  PCMSK2 |= B00000001;  //A8 will trigger  pin changed  interrupt
  PCMSK2 |= B00000010;  //A9 will trigger  pin changed  interrupt
  initMpu();

  avgHeadStart = millis();

  Stop();

}

// ====== Loop ======
void loop() {

  //stable (but slow) orientation using magnimeter 
  if (millis() - headStart >= 125) 
  {
    sum += getHeading();
    counter++;
    //heading = getHeading();
    headStart = millis();
    if (heading < 0) heading += 360.0;
  }
  if (millis() - avgHeadStart >= 2000) 
  {
     heading = sum / counter;
     sum = 0;
     counter = 0;
     if (heading < 0) heading += 360.0;
     avgHeadStart = millis();
   }
  
  //zAngle using gyro
  if ((millis() - yawAngleStart) > 40) {
    // get data every 10ms
    currentAngle = getYawGyro();
    yawAngleStart = millis();
    distance = distanceStep * (pulsesR*mRDirection + pulsesL*mLDirection) / 2.0;
    //angleEncoder += (distanceStep*(pulsesR*mRDirection - pulsesL*mLDirection) / 2.0) /CARRadius*180/PI;
    pulsesR = 0; pulsesL = 0;
    xPosition += distance * cos(currentAngle / 180 * PI);
    yPosition += distance * sin(currentAngle / 180 * PI);
    //Serial.print(currentAngle);

    data = String(currentAngle) + "*"+ String(xPosition) +"*"+ String(yPosition) + "*"+ String((int)objectDetected);    
    Serial1.println(data);
    //Serial.println(data);
  }
  
if (isAuto)
{
  switch (state)
  {
    case straight:
      if(!LineEndReached()) 
          MoveStraight();
      else
          supposedAngle = currentAngle + rotSign*90;
      break;
    case turning:   
      PerformRotation();
      if(RotationFinished())
      {
        Orientation temp = curOrient;
        if(curOrient.axis == Y)
          { curOrient.axis = X; curOrient.sign = -prevOrient.sign; currentHLine++;}
        else 
          curOrient = prevOrient;
          
        prevOrient = temp;
        
        state = straight;
        
      }
        break;

    case bypassTarget:
    break;
    case paused:
    Stop(); break;

    case finished:
      Stop(); break;
  }
}


  //metal sensor detection with filter
  if (digitalRead(sensorIn) == HIGH) 
  {
    if (!detectStart) 
    {
      detectStart = true;
      detectionStartTime = millis();
    }
    //cancel false detection
    if (millis() - detectionStartTime > 500) 
    {
      digitalWrite(sensorOut, HIGH);
      objectDetected = true;

    }
  } 
  else 
  {
    digitalWrite(sensorOut, LOW);
    detectStart = false;
    objectDetected = false;
  }

  if (Serial1.available()) 
  {
    //String data = Serial1.readStringUntil('\n');
    //Serial.println(data);
    String data = Serial1.readStringUntil('\n');
    Serial.println(data);
    updateState(data);
    manualStart = millis();
  }
  
}

void updateState(String data) 
{
    char cmd = data[0];

    switch(cmd)
    {
      case 'a':
      isAuto = true; break;
      case 'm':
      isAuto = false; break;
      case 'R':
      Reset(); break;
      case 'f':
      Forward(speed); break;
      case 'b':
      Backward(speed); break;
      case 'l': 
      RotateLeft(sp_rotate); break;
      case 'r':
      RotateRight(sp_rotate); break; 
      case 's':
      Stop(); break;
      case 'u':
      speedup(); break;   
      case 'd':
      speeddown(); break; 
      case 'w':
      sweepWidth=data.substring(2).toFloat(); break; 
      case 'h':
      sweepHeight=data.substring(2).toFloat(); break;
      case 'W':
      carWidth=data.substring(2).toFloat(); break;
      case 'L':
      lineWidth=data.substring(2).toFloat(); 
      linesNum = sweepHeight/lineWidth; break;
      case 'p': 
      if(state == paused)     
      {state = lastState; lastState=paused;}
      else
      {lastState = state; state = paused;  }   
      break;
    }
}
void Reset()
{
    Stop();
    state = straight;
    yawOffset = getYawGyro();
    currentAngle = getYawGyro();
    xPosition = 0;
    yPosition = 0;
    heading = 0;    
    isAuto = false;
    pulsesL = pulsesR = 0;
    currentHLine = 0;
    linesNum = sweepHeight/lineWidth;
    
    curOrient.axis = X;
    curOrient.sign = 1;
    prevOrient.axis = Y;
    prevOrient.sign = 1;
}


void MoveStraight()
{

  Forward(speed);
  //modifiy directions angles
  //  if(curOrient.axis == X && curOrient.sign == 1)
  //  {
  //     if(currentAngle > 2)
  //     ForwardZigZag(75, 150); //R,L zigzag right
  //     else if(currentAngle < 358) 
  //     ForwardZigZag(150, 75); //zigzag lift
  //     else 
  //     Forward (150);      
  //  }
  //  if(curOrient.axis == X && curOrient.sign == -1)
  //  {
  //     if(currentAngle > 182)
  //     ForwardZigZag(75, 150); //R,L zigzag right
  //     else if(currentAngle < 178) 
  //     ForwardZigZag(150, 75); //zigzag lift
  //     else 
  //     Forward (150);      
  //  }
  //  if(curOrient.axis == Y)
  //  {
  //     if(currentAngle > 95)
  //     ForwardZigZag(75, 150); //R,L zigzag right
  //     else if(currentAngle < 85) 
  //     ForwardZigZag(150, 75); //zigzag lift
  //     else 
  //     Forward (150);      
  //  }

}

bool LineEndReached()
{
   //if(currentHLine%2==0 && xPosition >= sweepWidth) //end of even lines move in +x direction
   if(curOrient.axis==X) 
   if (curOrient.sign ==1 && xPosition >= sweepWidth || curOrient.sign ==-1 && xPosition <= 0) 
   {
      if(currentHLine == linesNum-1)
      {
        state = finished;
        return true;
      }
      rotSign = curOrient.sign; //rotation sign equal x direction
      state = turning;
      return true;
   }
   if(curOrient.axis==Y && yPosition >= (currentHLine+1)*lineWidth) //end of even lines move in +x direction
   {
      rotSign = prevOrient.sign;  //rotation sign equal x direction
      state = turning;
      return true;
   }

  return false;
}
void PerformRotation()
{
  if(rotSign>0)
   RotateLeft(sp_rotate);
  else
   RotateRight(sp_rotate);
}
bool RotationFinished()
{
  if( abs(supposedAngle-currentAngle) <= 1)
  return true;

  return false;
}

// ====== Ultrasonic ======
long readUltrasonic(int trigPin, int echoPin) 
{
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  return pulseIn(echoPin, HIGH) / 58.0;  // cm
}
void Forward(int speed) 
{
  mLDirection = 1; mRDirection = 1;
  digitalWrite(MFL1, HIGH);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, HIGH);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, HIGH);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, HIGH);  digitalWrite(MBR2, LOW);

  analogWrite(MFLSP, speed);  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);  analogWrite(MBRSP, speed);  
}
void ForwardZigZag(int speedRight, int speedLeft) 
{
  mLDirection = 1; mRDirection = 1;
  digitalWrite(MFL1, HIGH);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, HIGH);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, HIGH);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, HIGH);  digitalWrite(MBR2, LOW);

  analogWrite(MFLSP, speedRight);  analogWrite(MFRSP, speedLeft);
  analogWrite(MBLSP, speedRight);  analogWrite(MBRSP, speedLeft);  
}
void Backward(int speed) 
{
  mLDirection = -1; mRDirection = -1;
  digitalWrite(MFL1, LOW);  digitalWrite(MFL2, HIGH);
  digitalWrite(MFR1, LOW);  digitalWrite(MFR2, HIGH);
  digitalWrite(MBL1, LOW);  digitalWrite(MBL2, HIGH);
  digitalWrite(MBR1, LOW);  digitalWrite(MBR2, HIGH);

  analogWrite(MFLSP, speed);  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);  analogWrite(MBRSP, speed);  
}
void Stop() {
  mLDirection = 1; mRDirection = 1;
  digitalWrite(MFL1, LOW);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, LOW);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, LOW);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, LOW);  digitalWrite(MBR2, LOW);
}
void RotateLeft(int speed) 
{
  mLDirection = -1; mRDirection = 1;  
  digitalWrite(MFL1, LOW);  digitalWrite(MFL2, HIGH);
  digitalWrite(MFR1, HIGH);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, LOW);  digitalWrite(MBL2, HIGH);
  digitalWrite(MBR1, HIGH);  digitalWrite(MBR2, LOW);

  analogWrite(MFLSP, speed);  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);  analogWrite(MBRSP, speed);  
}
void RotateRight(int speed) 
{
  mLDirection = 1; mRDirection = -1;
  digitalWrite(MFL1, HIGH);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, LOW);  digitalWrite(MFR2, HIGH);
  digitalWrite(MBL1, HIGH);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, LOW);  digitalWrite(MBR2, HIGH);

  analogWrite(MFLSP, speed);  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);  analogWrite(MBRSP, speed);  
}
void CurveLeft() 
{
  mLDirection = -1; mRDirection = 1;
  digitalWrite(MFL1, LOW);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, HIGH);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, LOW);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, HIGH);  digitalWrite(MBR2, LOW);
}
void CurveRight() 
{
  mLDirection = 1; mRDirection = -1;
  digitalWrite(MFL1, HIGH);  digitalWrite(MFL2, LOW);
  digitalWrite(MFR1, LOW);  digitalWrite(MFR2, LOW);
  digitalWrite(MBL1, HIGH);  digitalWrite(MBL2, LOW);
  digitalWrite(MBR1, LOW);  digitalWrite(MBR2, LOW);
}
void speedup() 
{
  speed += 5;
  analogWrite(MFLSP, speed);
  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);
  analogWrite(MBRSP, speed);
}
void speeddown() 
{
  speed -= 5;
  analogWrite(MFLSP, speed);
  analogWrite(MFRSP, speed);
  analogWrite(MBLSP, speed);
  analogWrite(MBRSP, speed);
}



bool A8_state = LOW;
bool A9_state = LOW;
ISR(PCINT2_vect)
{
  if (digitalRead(A8) && !A8_state) {
    A8_state = HIGH;
    pulsesL++;

  } else if (!digitalRead(A8) && A8_state) {
    A8_state = LOW;
  }
  if (digitalRead(A9) && !A9_state) {
    A9_state = HIGH;
    pulsesR++;
  } else if (!digitalRead(A9) && A9_state) {
    A9_state = LOW;
  }
}

float getYawGyro() 
{
  const float gyroSensitivity = 131.0;  // ثابت يدوي بدل myIMU.gRes

  myIMU.readGyroData(myIMU.gyroCount);
  // إزالة البايس + التحويل إلى deg/s
  float gz = ((float)myIMU.gyroCount[2] - gyroBiasZ) / gyroSensitivity;

  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;  // بالثانية
  lastTime = now;

  yaw += gz * dt;  // تكامل السرعة الزاوية

  if (yaw < 0) yaw += 360;
  if (yaw >= 360) yaw -= 360;

  // إعادة الضبط بناءً على الإزاحة
  float adjustedYaw = yaw - yawOffset;
  if (adjustedYaw < 0) adjustedYaw += 360;
  if (adjustedYaw >= 360) adjustedYaw -= 360;

  return adjustedYaw;
}

// --------- GET HEADING MAG ---------
float getHeading() 
{
  myIMU.readMagData(myIMU.magCount);

  float mx = (float)myIMU.magCount[0] * magCalibration[0];
  float my = (float)myIMU.magCount[1] * magCalibration[1];
  float mz = (float)myIMU.magCount[2] * magCalibration[2];

  float heading = atan2(my, mx) * 180.0 / PI;
  return heading;
}

void initMpu() 
{
  delay(2000);

  //Serial.println("Starting MPU9250...");
  byte c = myIMU.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
  //Serial.print("WHO_AM_I: 0x");
  //Serial.println(c, HEX);

  if (c == 0x73 || c == 0x71) {  // يقبل القيمتين
    //Serial.println("MPU9250 (or compatible) detected!");
    myIMU.initMPU9250();
    myIMU.initAK8963(magCalibration);

    // ---- معايرة الجايرو Z ----
    long sum = 0;
    const int N = 200;
    //Serial.println("Calibrating gyro Z bias... Keep sensor still!");
    for (int i = 0; i < N; i++) {
      myIMU.readGyroData(myIMU.gyroCount);
      sum += myIMU.gyroCount[2];
      delay(5);
    }
    gyroBiasZ = (float)sum / N;
    //Serial.print("Gyro Z bias: ");
    //Serial.println(gyroBiasZ);

    lastTime = millis();
  } 
  else 
  {
    //Serial.println("Unsupported sensor detected!");
    //while (1)
    //  ;
  }
}