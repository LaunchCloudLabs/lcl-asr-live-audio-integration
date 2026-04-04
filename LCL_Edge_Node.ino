/*
 * LCL Sovereign Edge Node V14 - "The VU Pumper"
 * LED VU meter: fast flash = loud, slow flash = quiet
 * I2S INMP441 only (GPIO 14/15/4). No analogRead - Core 3.x driver_ng conflict.
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>

const char* ssid     = "Verizon_SD4YSF";
const char* password = "ply6glee9ran";
const char* ws_host  = "192.168.1.160";
const int   ws_port  = 8765;

#define LED_PIN     2
#define BUF_SAMPLES 512

WebSocketsClient webSocket;
bool hubConnected = false;

unsigned long lastLedToggle = 0;
bool ledState = false;
int ledInterval = 0;

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) {
    hubConnected = true;
    Serial.println("HUB SYNCED");
    webSocket.sendTXT("{\"type\":\"esp32_hello\"}");
  } else if (type == WStype_DISCONNECTED) {
    hubConnected = false;
    Serial.println("HUB LOST");
  }
}

float calcRMS(int16_t* buf, int len) {
  float sum = 0;
  for (int i = 0; i < len; i++) {
    float s = buf[i] / 32768.0f;
    sum += s * s;
  }
  return sqrtf(sum / len);
}

void updateLED() {
  if (ledInterval <= 0) { digitalWrite(LED_PIN, LOW); return; }
  unsigned long now = millis();
  if (now - lastLedToggle >= (unsigned long)ledInterval) {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState ? HIGH : LOW);
    lastLedToggle = now;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.print("WiFi connecting");
  WiFi.begin(ssid, password);
  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    digitalWrite(LED_PIN, dots++ % 2);
    Serial.print(".");
  }
  digitalWrite(LED_PIN, LOW);
  Serial.println("\nWiFi OK: " + WiFi.localIP().toString());

  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = 16000,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = 0,
    .dma_buf_count        = 4,
    .dma_buf_len          = BUF_SAMPLES,
    .use_apll             = false
  };
  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = 14,
    .ws_io_num    = 15,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = 4
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
  Serial.println("I2S OK");

  webSocket.begin(ws_host, ws_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);
  Serial.println("V14 PUMPER READY.");
}

void loop() {
  webSocket.loop();
  updateLED();

  int16_t buf[BUF_SAMPLES];
  size_t bytesRead = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, buf, sizeof(buf), &bytesRead, pdMS_TO_TICKS(50));

  if (err != ESP_OK || bytesRead == 0) {
    Serial.println("[I2S TIMEOUT] No data from mic");
    return;
  }

  float rms = calcRMS(buf, bytesRead / 2);
  Serial.printf("RMS=%.5f HUB=%s\n", rms, hubConnected ? "OK" : "OFF");

  if (rms < 0.001f)       ledInterval = 0;
  else if (rms >= 0.05f)  ledInterval = 40;
  else {
    float norm = (rms - 0.001f) / 0.049f;
    ledInterval = (int)(400.0f - norm * 360.0f);
  }

  if (hubConnected) webSocket.sendBIN((uint8_t*)buf, bytesRead);
}
