import requests

def get_sky_data():
    # The 'states/all' endpoint provides data for all active aircraft
    url = "https://opensky-network.org/api/states/all"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Check for HTTP errors
        
        data = response.json()
        states = data.get('states', [])

        print(f"Total flights found: {len(states)}\n")
        print(f"{'ICAO24':<10} | {'Callsign':<10} | {'Country':<15} | {'Altitude (m)':<12} | {'Velocity (m/s)'}")
        print("-" * 75)

        # Print the first 10 flights as an example
        for flight in states[:10]:
            icao24 = flight[0]
            callsign = flight[1].strip() if flight[1] else "N/A"
            country = flight[2]
            altitude = flight[7] if flight[7] else 0
            velocity = flight[9] if flight[9] else 0
            
            print(f"{icao24:<10} | {callsign:<10} | {country:<15} | {altitude:<12} | {velocity}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_sky_data()