import numpy as np

"""
Airport Database Documentation


Airport ID	Unique OpenFlights identifier for this airport.
Name	Name of airport. May or may not contain the City name.
City	Main city served by airport. May be spelled differently from Name.
Country	Country or territory where airport is located. See countries.dat to cross-reference to ISO 3166-1 codes.
IATA	3-letter IATA code. Null if not assigned/unknown.
ICAO	4-letter ICAO code.
Null if not assigned.
Latitude	Decimal degrees, usually to six significant digits. Negative is South, positive is North.
Longitude	Decimal degrees, usually to six significant digits. Negative is West, positive is East.
Altitude	In feet.
Timezone	Hours offset from UTC. Fractional hours are expressed as decimals, eg. India is 5.5.
DST	Daylight savings time. One of E (Europe), A (US/Canada), S (South America), O (Australia), Z (New Zealand), N (None) or U (Unknown). See also: Help: Time
Tz database time zone	Timezone in "tz" (Olson) format, eg. "America/Los_Angeles".
Type	Type of the airport. Value "airport" for air terminals, "station" for train stations, "port" for ferry terminals and "unknown" if not known. In airports.csv, only type=airport is included.
Source	Source of this data. "OurAirports" for data sourced from OurAirports, "Legacy" for old data not matched to OurAirports (mostly DAFIF), "User" for unverified user contributions. In airports.csv, only source=OurAirports is included.

"""


"""
Route Database Documentation

Airline	2-letter (IATA) or 3-letter (ICAO) code of the airline.
Airline ID	Unique OpenFlights identifier for airline (see Airline).
Source airport	3-letter (IATA) or 4-letter (ICAO) code of the source airport.
Source airport ID	Unique OpenFlights identifier for source airport (see Airport)
Destination airport	3-letter (IATA) or 4-letter (ICAO) code of the destination airport.
Destination airport ID	Unique OpenFlights identifier for destination airport (see Airport)
Codeshare	"Y" if this flight is a codeshare (that is, not operated by Airline, but another carrier), empty otherwise.
Stops	Number of stops on this flight ("0" for direct)
Equipment	3-letter codes for plane type(s) generally used on this flight, separated by spaces

"""


def parse_airports():
    airports = dict()

    with open('data/airports.dat') as f:
        for line in f:
            data = line.split(',')
            country = data[3].replace("\"", "")
            if country == "United States" or country == "USA" or country == "United States of America":
                airports[int(data[0])] = [data[1].replace("\"", ""), # Name
                  data[2].replace("\"", ""), # City
                  data[4].replace("\"", ""), # IATA
                  data[5].replace("\"", ""), # ICAO
                  [float(data[6]), float(data[7])], 0]
    return airports


def parse_routes(pre_data):
    new_data = pre_data

    with open('data/routes.dat') as f:
        for line in f:
            data = line.split(',')

            if data[3] != '\N' and int(data[3]) in pre_data:
                new_data[int(data[3])][5] = new_data[int(data[3])][5] + 1
            if data[5] != '\N' and int(data[5]) in pre_data:
                new_data[int(data[5])][5] = new_data[int(data[5])][5] + 1

    return new_data


if __name__ == "__main__":
    """
    Parses airport database into dictionary key'd with Openflights ID storing

        1. Airport name
        2. Airport city
        3. IATA code of airport
        4. ICAO code of airport
        5. Lat, Long coordinates of airport 
        6. Node weight of airport (Based on number of routes to/from airport, but city population might be a better proxy)
    """

    data = parse_airports()
    data = parse_routes(data)

    print len(data.keys())

    print "LAX Ranking: ",          data[3484][5]
    print "Burbank Ranking: ",      data[3644][5]
    print "JFK Ranking: ",          data[3797][5]
    print "La Guardia Ranking: ",   data[3697][5]

    np.save('parsed_data.npy', data)