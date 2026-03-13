#include <Adafruit_MPR121.h>
#include <Wire.h>

#ifndef _BV
#define _BV(bit) (1 << (bit))
#endif

Adafruit_MPR121 cap = Adafruit_MPR121();

const int NUM_SENSORS = 9;  // ELE0–ELE8 (A1-A3, B1-B3, C1-C3)
const int WINDOW_SIZE = 10;
int history[NUM_SENSORS][WINDOW_SIZE];
int readIndex = 0;

void setup() {
  Serial.begin(115200);

  if (!cap.begin(0x5A)) {
    Serial.println("MPR121 not found!");
    while (1);
  }

  // --- CUSTOM CONFIGURATION ---
  cap.writeRegister(0x5E, 0x00);  // Stop mode (required before config changes)

  // Auto-config
  cap.writeRegister(0x7B, 0x00);  // Auto-Config Control 0
  cap.writeRegister(0x7D, 0xE4);  // USL
  cap.writeRegister(0x7E, 0x94);  // LSL
  cap.writeRegister(0x7F, 0xC8);  // Target Level

  // Global CDC/CDT
  cap.writeRegister(0x5C, 0x16);  // charge current
  cap.writeRegister(0x5D, 0x60);  // charge time

  // Drift fix — disable downward baseline drift, set fast recovery (rising + falling)
  cap.writeRegister(0x2B, 0x01);  // MHD Rising
  cap.writeRegister(0x2C, 0x01);  // NHD Rising
  cap.writeRegister(0x2D, 0x00);  // NCL Rising  (no downward drift)
  cap.writeRegister(0x2E, 0x00);  // FDL Rising
  cap.writeRegister(0x2F, 0x01);  // MHD Falling (fast recovery)
  cap.writeRegister(0x30, 0x01);  // NHD Falling

  // Run mode: enable NUM_SENSORS electrodes (ELE0 through ELE8)
  cap.writeRegister(0x5E, 0x80 | NUM_SENSORS);
  // --- END CUSTOM CONFIGURATION ---

  // Prime the smoothing buffer with real sensor readings
  // so averages are valid from the very first loop iteration
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
      history[sensor][i] = cap.filteredData(sensor);
    }
  }
}

void loop() {
  // 1. Store latest raw readings into circular buffer
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    history[sensor][readIndex] = cap.filteredData(sensor);
  }
  readIndex = (readIndex + 1) % WINDOW_SIZE;

  // 2. Print smoothed averages as CSV
  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    long sum = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      sum += history[sensor][i];
    }
    Serial.print(sum / WINDOW_SIZE);
    if (sensor < NUM_SENSORS - 1) {
      Serial.print(",");
    }
  }

  // 3. Trailing comma for empty "Note" column, then newline
  Serial.print(",");
  Serial.println();

  delay(20);
}