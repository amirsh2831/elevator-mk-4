#include <Arduino.h>
#include <esp_heap_caps.h>
#include "driver/i2s.h"
#include "keyword_weights.h"
#include "digit_weights.h"

// The two headers must contain the renamed exports described in the project plan.
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK 5
#define I2S_WS 7
#define I2S_DATA 9
#define WAKE_RECORD_LED 16
#define NUMBER_RECORD_LED 15
#define SAMPLE_RATE 44100
#define FRAME_SIZE 1024
#define HOP_SIZE 512
#define VAD_THRESHOLD 0.05f
#define WAKE_SECONDS 2
#define NUMBER_SECONDS 2

#define KEY_INPUT 237
#define DIGIT_INPUT 84
#define AUDIO_SAMPLES (SAMPLE_RATE * WAKE_SECONDS)
#define MAX_SOURCE_FRAMES ((AUDIO_SAMPLES - FRAME_SIZE) / HOP_SIZE + 1)

// Large buffers and model data are placed in PSRAM.  The header arrays remain
// the source data in flash; these copies are the data used during inference.
static int16_t *audio = nullptr;
static float *frameEnergy = nullptr;
// A two-second capture can contain more source frames than the 79-frame
// keyword target.  Keep the source buffer large; it is resized afterward.
static float (*features)[3] = nullptr;
static float input[KEY_INPUT];
static float hiddenA[128], hiddenB[64];

static float *keywordW0, *keywordB0, *keywordW1, *keywordB1, *keywordW2, *keywordB2;
static float *keywordMean, *keywordScale;
static float *digitW0, *digitB0, *digitW1, *digitB1, *digitW2, *digitB2;
static float *digitMean, *digitScale;

static void *psramArray(size_t bytes) {
  return ps_malloc(bytes);
}

static bool allocateRuntimeData() {
  audio = (int16_t *)psramArray(sizeof(int16_t) * AUDIO_SAMPLES);
  frameEnergy = (float *)psramArray(sizeof(float) * MAX_SOURCE_FRAMES);
  features = (float (*)[3])psramArray(sizeof(float) * MAX_SOURCE_FRAMES * 3);

  keywordW0 = (float *)psramArray(sizeof(float) * 30336);
  keywordB0 = (float *)psramArray(sizeof(float) * 128);
  keywordW1 = (float *)psramArray(sizeof(float) * 8192);
  keywordB1 = (float *)psramArray(sizeof(float) * 64);
  keywordW2 = (float *)psramArray(sizeof(float) * 128);
  keywordB2 = (float *)psramArray(sizeof(float) * 2);
  keywordMean = (float *)psramArray(sizeof(float) * KEY_INPUT);
  keywordScale = (float *)psramArray(sizeof(float) * KEY_INPUT);

  digitW0 = (float *)psramArray(sizeof(float) * 4032);
  digitB0 = (float *)psramArray(sizeof(float) * 48);
  digitW1 = (float *)psramArray(sizeof(float) * 1152);
  digitB1 = (float *)psramArray(sizeof(float) * 24);
  digitW2 = (float *)psramArray(sizeof(float) * 240);
  digitB2 = (float *)psramArray(sizeof(float) * 10);
  digitMean = (float *)psramArray(sizeof(float) * DIGIT_INPUT);
  digitScale = (float *)psramArray(sizeof(float) * DIGIT_INPUT);

  return audio && frameEnergy && features && keywordW0 && keywordB0 && keywordW1 && keywordB1 &&
         keywordW2 && keywordB2 && keywordMean && keywordScale && digitW0 && digitB0 && digitW1 &&
         digitB1 && digitW2 && digitB2 && digitMean && digitScale;
}

static void copyRuntimeData() {
  memcpy(keywordW0, keyword_network_0_weight, sizeof(float) * 30336);
  memcpy(keywordB0, keyword_network_0_bias, sizeof(float) * 128);
  memcpy(keywordW1, keyword_network_3_weight, sizeof(float) * 8192);
  memcpy(keywordB1, keyword_network_3_bias, sizeof(float) * 64);
  memcpy(keywordW2, keyword_network_6_weight, sizeof(float) * 128);
  memcpy(keywordB2, keyword_network_6_bias, sizeof(float) * 2);
  memcpy(keywordMean, keyword_scaler_mean, sizeof(float) * KEY_INPUT);
  memcpy(keywordScale, keyword_scaler_scale, sizeof(float) * KEY_INPUT);
  memcpy(digitW0, digit_network_0_weight, sizeof(float) * 4032);
  memcpy(digitB0, digit_network_0_bias, sizeof(float) * 48);
  memcpy(digitW1, digit_network_3_weight, sizeof(float) * 1152);
  memcpy(digitB1, digit_network_3_bias, sizeof(float) * 24);
  memcpy(digitW2, digit_network_6_weight, sizeof(float) * 240);
  memcpy(digitB2, digit_network_6_bias, sizeof(float) * 10);
  memcpy(digitMean, digit_scaler_mean, sizeof(float) * DIGIT_INPUT);
  memcpy(digitScale, digit_scaler_scale, sizeof(float) * DIGIT_INPUT);
}

static void setupI2S() {
  i2s_config_t c = {};
  c.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  c.sample_rate = SAMPLE_RATE;
  c.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  c.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  c.communication_format = I2S_COMM_FORMAT_I2S_MSB;
  c.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  c.dma_buf_count = 6; c.dma_buf_len = 512;
  c.use_apll = true;
  i2s_pin_config_t p = { I2S_BCLK, I2S_PIN_NO_CHANGE, I2S_WS, I2S_DATA };
  i2s_driver_install(I2S_PORT, &c, 0, nullptr);
  i2s_set_pin(I2S_PORT, &p);
  i2s_zero_dma_buffer(I2S_PORT);
}

static void recordAudio(uint32_t seconds, int16_t *dst) {
  uint32_t wanted = min((uint32_t)(SAMPLE_RATE * seconds),
                         (uint32_t)AUDIO_SAMPLES);
  uint32_t got = 0;
  int32_t buf[128];
  while (got < wanted) {
    size_t bytes = 0;
    if (i2s_read(I2S_PORT, buf, sizeof(buf), &bytes, portMAX_DELAY) != ESP_OK) continue;
    uint32_t words = bytes / sizeof(int32_t);
    // RIGHT_LEFT interleaves left, right. The microphone is configured LEFT.
    for (uint32_t i = 0; i < words && got < wanted; i += 2){

    
      int32_t converted = buf[i] >> 8;

    if (converted > 32767)
        converted = 32767;

    if (converted < -32768)
        converted = -32768;

    dst[got++] = (int16_t)converted;
    }
  }
}

static int activeFrames(uint32_t samples) {
  int count = samples >= FRAME_SIZE ? (samples - FRAME_SIZE) / HOP_SIZE + 1 : 0;
  if (!count) return 0;
  float maximum = 0;
  for (int n = 0; n < count; n++) {
    float e = 0;
    int start = n * HOP_SIZE;
    for (int j = 0; j < FRAME_SIZE; j++) { float x = audio[start+j] / 32768.0f; e += x*x; }
    frameEnergy[n] = e; if (e > maximum) maximum = e;
  }
  if (maximum <= 0) return 0;
  int first = 0, last = count - 1;
  while (first < count && frameEnergy[first] / maximum < VAD_THRESHOLD) first++;
  while (last >= 0 && frameEnergy[last] / maximum < VAD_THRESHOLD) last--;
  return last >= first ? (last - first) : 0; // matches Python's w_end-w_start
}

static void resizeAndFlatten(int sourceCount, int target, bool keyword) {
  if (sourceCount <= 0) { for (int i=0;i<target*3;i++) input[i]=0; return; }
  for (int i = 0; i < target; i++) {
    float pos = target > 1 ? (float)i * (sourceCount - 1) / (target - 1) : 0;
    int j = (int)pos; float w = pos - j; if (j >= sourceCount-1) j = sourceCount-1, w=0;
    for (int k=0;k<3;k++) {
      float v = features[j][k]; if (w && j+1<sourceCount) v = v*(1-w)+features[j+1][k]*w;
      input[i*3+k] = v;
    }
  }
  (void)keyword;
}

static int makeFeatures(uint32_t samples, bool keyword) {
  int count = activeFrames(samples); int first = 0;
  if (!count) return 0;
  float maxE=0,maxA=0,maxZ=0;
  // Re-find the first active frame (activeFrames deliberately returns Python length).
  int total=(samples-FRAME_SIZE)/HOP_SIZE+1;
  float peak=0;

  for(int n=0;n<total;n++) if(frameEnergy[n]>peak) peak=frameEnergy[n];
  while(first<total && frameEnergy[first]/peak<VAD_THRESHOLD) first++;

  for(int n=0;n<count;n++) {
    int s=(first+n)*HOP_SIZE;
    float e=0,a=0; int z=0;
    float prev=audio[s]/32768.0f; for(int j=0;j<FRAME_SIZE;j++){float x=audio[s+j]/32768.0f;e+=x*x;a+=fabsf(x);if(j&&x*prev<0)z++;prev=x;}
    features[n][0]=e; features[n][1]=a/FRAME_SIZE; features[n][2]=(float)z; if(e>maxE)maxE=e;if(a>maxA)maxA=a;if(z>maxZ)maxZ=z;
  }
  if(keyword) for(int n=0;n<count;n++){features[n][0]/=maxE;features[n][1]/=maxA;features[n][2]/=maxZ;}
  resizeAndFlatten(count, keyword?79:28, keyword); return keyword?KEY_INPUT:DIGIT_INPUT;
}

static int classify(float *x, int in, bool keyword) {
  if(keyword){for(int i=0;i<in;i++) input[i]=(x[i]-keywordMean[i])*keywordScale[i];
    for(int j=0;j<128;j++){
      float s=keywordB0[j];for(int i=0;i<237;i++)s+=input[i]*keywordW0[j*237+i];
      hiddenA[j]=s>0?s:0;
    }

    for(int j=0;j<64;j++){
      float s=keywordB1[j];
      for(int i=0;i<128;i++)s+=hiddenA[i]*keywordW1[j*128+i];
      hiddenB[j]=s>0?s:0;
    }

    int best=0;
    float bv=-1e30;
    for(int j=0;j<2;j++){
      float s=keywordB2[j];
      for(int i=0;i<64;i++)s+=hiddenB[i]*keywordW2[j*64+i];
      if(s>bv)bv=s,best=j;
    }

    return best;
  } else {
    for(int i=0;i<in;i++)input[i]=(x[i]-digitMean[i])*digitScale[i];

    for(int j=0;j<48;j++){
      float s=digitB0[j];
      for(int i=0;i<84;i++)s+=input[i]*digitW0[j*84+i];
      hiddenA[j]=s>0?s:0;
    }

    for(int j=0;j<24;j++){
      float s=digitB1[j];
      for(int i=0;i<48;i++)s+=hiddenA[i]*digitW1[j*48+i];
      hiddenB[j]=s>0?s:0;
    }

    int best=0;
    float bv=-1e30;
    for(int j=0;j<10;j++){
      float s=digitB2[j];
      for(int i=0;i<24;i++)s+=hiddenB[i]*digitW2[j*24+i];
      if(s>bv)bv=s,best=j;
    }

    return best;
  }
}

void setup(){
  Serial.begin(115200);
  pinMode(WAKE_RECORD_LED,OUTPUT);
  pinMode(NUMBER_RECORD_LED,OUTPUT);
  Serial.printf("PSRAM found: %s, total: %u, free: %u\n", psramFound()?"yes":"no", ESP.getPsramSize(), ESP.getFreePsram());

  if(!psramFound() || !allocateRuntimeData()){
    Serial.println("ERROR: PSRAM allocation failed");
    while(true) delay(1000);
  }

  copyRuntimeData();
  Serial.printf("PSRAM runtime data ready; free: %u\n", ESP.getFreePsram());
  setupI2S(); Serial.println("Inference ready");
}

void loop(){
  Serial.println("Recording wake word...");
  digitalWrite(WAKE_RECORD_LED,HIGH);
  recordAudio(WAKE_SECONDS,audio);
  digitalWrite(WAKE_RECORD_LED,LOW);
  makeFeatures(SAMPLE_RATE*WAKE_SECONDS,true);
  int key=classify(input,KEY_INPUT,true);
  Serial.printf("Keyword class: %d\n",key);

  if(key!=1){
    Serial.println("Wake word not detected;
      retrying.");
      return;
  }

  Serial.println("Wake word detected. Recording number...");
  digitalWrite(NUMBER_RECORD_LED,HIGH);
  recordAudio(NUMBER_SECONDS,audio);
  digitalWrite(NUMBER_RECORD_LED,LOW);
  makeFeatures(SAMPLE_RATE*NUMBER_SECONDS,false);
  int digit=classify(input,DIGIT_INPUT,false);
  Serial.printf("Final number: %d\n",digit+1);
  delay(250);
}
