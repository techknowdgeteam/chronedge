# regulator.py  ← FINAL SCRIPT WITH CATEGORIZED OUTPUT

import ctypes
from ctypes import wintypes
import serial.tools.list_ports 

# --- WMI for General Hardware (Windows Only) ---
try:
    import wmi
    # Initialize WMI connection
    c = wmi.WMI()
except ImportError:
    print("Error: The 'wmi' library is missing. Run 'python -m pip install wmi'")
    c = None
except Exception as e:
    print(f"Error initializing WMI: {e}")
    c = None

# --- Core Functions ---

def get_system_status():
    """Retrieves battery percentage, charging status, and charging controllers."""
    
    # 1. Get Battery Status (Charging: Yes/No)
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", wintypes.BYTE),
                    ("BatteryFlag", wintypes.BYTE),
                    ("BatteryLifePercent", wintypes.BYTE),
                    ("Reserved1", wintypes.BYTE),
                    ("BatteryLifeTime", wintypes.DWORD),
                    ("BatteryFullLifeTime", wintypes.DWORD)]

    sps = SPS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps))
    
    # Determine Charging Status (1=Plugged in/Charging, 0=On battery)
    is_charging = sps.ACLineStatus == 1
    
    # Format the Battery Percentage
    pct = sps.BatteryLifePercent if sps.BatteryLifePercent <= 100 else "Unknown"
    
    # 2. Get Power Control Hardware (Internal Controllers)
    power_controllers = []
    if c:
        try:
            # Query for devices related to power management and AC adapter
            query_devices = c.query("SELECT * FROM Win32_PnPEntity WHERE Caption LIKE '%AC Adapter%' OR Caption LIKE '%Battery%'")
            for device in query_devices:
                power_controllers.append(device.Caption)
        except Exception as e:
            power_controllers.append(f"[Error querying WMI: {e}]")
    else:
        power_controllers.append("[WMI Not Available]")

    return {
        "percentage": pct,
        "charging_status": "Yes" if is_charging else "No",
        "power_controllers": power_controllers,
        "ac_line_status": "Plugged in" if is_charging else "Not plugged"
    }

def get_com_ports():
    """Lists available Serial COM ports (for completeness)."""
    ports = serial.tools.list_ports.comports()
    if not ports:  
        return "No COM ports detected (Serial/Arduino)"
    return "\n".join(f"{p.device.ljust(8)} - {p.description}" for p in ports)

# --- Execution and Formatted Output ---

status_data = get_system_status()

print("## 🔋 Battery and Charging Status")
print("-" * 35)
print(f"Battery Percentage: **{status_data['percentage']}%**")
print(f"Currently Charging: **{status_data['charging_status']}**")
print("-" * 35)

print("\n## ⚡ Power Port Analysis (Internal Controllers)")
print("*(Note: These are internal controllers, not physical port names like 'USB-C')*")
print("-" * 35)

# Find controllers relevant to charging (The 'Microsoft AC Adapter' is the key indicator)
found_charging_ports = False
for controller in status_data['power_controllers']:
    # We treat any device related to the AC adapter as a potential charging port interface
    if "AC Adapter" in controller or "Battery" in controller:
        found_charging_ports = True
        
        # This controller is the one that would indicate charging is happening
        if "AC Adapter" in controller:
            is_plugged = status_data['ac_line_status']
            currently_used = "**CURRENTLY IN USE**" if status_data['charging_status'] == "Yes" else "Not in use"
            
            print(f"**Port Interface: {controller}**")
            print(f"  - Charging Responsibility: **Yes**")
            print(f"  - Physical Status: {is_plugged}")
            print(f"  - Power Input State: {currently_used}")
        else:
            # Other power/battery controllers
            print(f"Port Interface: {controller}")
            print(f"  - Charging Responsibility: No (Monitors Battery)")
            print(f"  - Physical Status: Not applicable (Internal)")

if not found_charging_ports:
    print("No AC Adapter or Battery controllers detected via WMI.")

print("\n## 🔌 Available SERIAL COM Ports")
print("-" * 35)
print(get_com_ports())
