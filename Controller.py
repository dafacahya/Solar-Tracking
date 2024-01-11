import time
import threading
import RPi.GPIO as GPIO
from mpu6050 import mpu6050
from simple_pid import PID

# Tentukan nomor pin GPIO untuk masing-masing aktuator
aktuator_azimuth_pin = 17
aktuator_altitude_pin = 18

# Inisialisasi GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(aktuator_azimuth_pin, GPIO.OUT)
GPIO.setup(aktuator_altitude_pin, GPIO.OUT)

# Inisialisasi MPU-6050
sensor = mpu6050(0x68)  # Alamat I2C MPU-6050

# Inisialisasi PID untuk Azimuth dan Altitude
pid_azimuth = PID(1, 0.1, 0.05, setpoint=0)
pid_altitude = PID(1, 0.1, 0.05, setpoint=0)

# Fungsi untuk membaca data dari MPU-6050 dan menggerakkan aktuator
def monitor_mpu6050():
    try:
        while True:
            # Baca data dari MPU-6050
            accel_data, gyro_data = read_mpu6050()

            # Ambil data yang diperlukan (misalnya, gunakan gyro_data untuk kontrol PID)
            pid_input_azimuth = gyro_data['x']
            pid_input_altitude = gyro_data['y']

            # Hitung output PID untuk Azimuth dan Altitude
            pid_output_azimuth = pid_azimuth(pid_input_azimuth)
            pid_output_altitude = pid_altitude(pid_input_altitude)

            # Gerakkan aktuator berdasarkan output PID
            move_actuators(pid_output_azimuth, pid_output_altitude)

            # Tunggu sejenak (sesuai dengan kebutuhan)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

# Fungsi untuk membaca data dari MPU-6050
def read_mpu6050():
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()
    return accel_data, gyro_data

# Fungsi untuk menggerakkan aktuator Azimuth dan Altitude berdasarkan data PID
def move_actuators(pid_output_azimuth, pid_output_altitude):
    # Implementasikan logika untuk menggerakkan aktuator berdasarkan PID output
    # Misalnya, gunakan PWM untuk mengendalikan kecepatan atau arah aktuator
    # Contoh: GPIO.output(aktuator_azimuth_pin, GPIO.HIGH) untuk mengaktifkan

# Inisialisasi dan jalankan thread pemantauan MPU-6050
mpu_thread = threading.Thread(target=monitor_mpu6050)
mpu_thread.daemon = True
mpu_thread.start()

# Jalankan program selama 5 menit
try:
    time.sleep(5 * 60)
except KeyboardInterrupt:
    pass

finally:
    # Bersihkan GPIO saat program dihentikan
    GPIO.cleanup()
