"""The madrigalWeb module provides access to all Madrigal data via web services."""

# $Id: madrigalWeb.py 7664 2024-07-30 14:39:18Z brideout $

# standard python imports
from dataclasses import dataclass
import os, os.path, sys
import traceback
import urllib.request, urllib.parse
import types
import re
import datetime
import requests

# third party imports
import packaging.version

# constants
TIMEOUT = 60 * 30  # timeout in seconds before skipping file
TIMEOUT2 = 60 * 3  # shorter time out


class MadrigalData:
    """MadrigalData is a class that acquires data from a particular Madrigal site.

    Usage example::

        import madrigalWeb.madrigalWeb
        test =  madrigalWeb.madrigalWeb.MadrigalData('http://madrigal.haystack.mit.edu')
        instList = test.getInstrumentList()

    Non-standard Python modules used: None

    Change history:

    Written by "Bill Rideout":mailto:wrideout@haystack.mit.edu  Feb. 10, 2004
    """

    def __init__(self, url: str):
        """__init__ initializes a MadrigalData object.

        Inputs::

            url - (str) url of main page of madrigal site. Example: 'http://madrigal.haystack.mit.edu'

        Affects: Converts main page to cgi url, and stores that.

        Also stores self.siteDict, with key = site id, value= site url, and self.siteId

        Exceptions: If url not found.
        """
        CGI_NAME = "accessData.cgi"
        CGI_ACCESS_DATA_URL_PATTERN = re.compile('".*' + CGI_NAME)

        # get base of url
        url_parts = urllib.parse.urlparse(url)
        url_base = url_parts[0] + "://" + url_parts[1]

        # read main url
        response = requests.get(url, timeout=TIMEOUT2)
        if response.status_code != 200:
            raise ValueError("unable to open url " + str(url))
        page = response.text
        result = CGI_ACCESS_DATA_URL_PATTERN.search(page)

        # check for success
        if result == None:
            raise ValueError("invalid url: " + str(url))

        result = result.group()
        if type(result) in (list, tuple):
            result = result[0]

        if len(result[1 : (-1 * len(CGI_NAME))]) < 2:
            self.cgiurl = url
            print(f"Warning: cgiurl is set to main url {self.cgiurl}, not cgi script")
        else:
            self.cgiurl = url_base + result[1 : (-1 * len(CGI_NAME))]

        if not self.cgiurl[-1] == "/":
            self.cgiurl += "/"

        self.site_dict = self.__getSiteDict()
        self.site_id = self.__getSiteId()
        self._madrigal_version = self.getVersion()

    def __getSiteDict(self):
        """__getSiteDict returns a dictionary with key = site id, value= site url.

        Uses getMetadata cgi script
        """
        url = urllib.parse.urljoin(self.cgiurl, "getMetadata?fileType=5")
        response = requests.get(url, timeout=TIMEOUT2)
        if response.status_code != 200:
            raise ValueError("unable to open url " + str(url))
        page = response.text

        lines = page.split("\n")
        site_dict = {}
        for line in lines:
            items = line.split(",")
            if len(items) < 4:
                continue
            site = int(items[0])
            site_url = f"http://{items[2]}/{items[3]}"
            site_dict[site] = site_url

        return site_dict

    def __getSiteId(self):
        """__getSiteId returns the local site id

        Uses getMetadata cgi script
        """
        url = urllib.parse.urljoin(self.cgiurl, "getMetadata?fileType=0")
        response = requests.get(url, timeout=TIMEOUT2)
        if response.status_code != 200:
            raise ValueError("unable to open url " + str(url))
        page = response.text

        lines = page.split("\n")
        for line in lines:
            items = line.split(",")
            if len(items) < 4:
                continue
            siteId = int(items[3])
            return siteId

        raise IOError("No siteId found")

    def getAllInstruments(self):
        """returns a list of all MadrigalInstruments at the given Madrigal site"""

        script_name = "getInstrumentsService.py"
        # url = self.cgiurl + scriptName
        url = urllib.parse.urljoin(self.cgiurl, script_name)
        response = requests.get(url, timeout=TIMEOUT2)
        if response.status_code != 200:
            raise ValueError("unable to open url " + str(url))
        page = response.text

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that html was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        for line in page:
            items = line.split(",")
            if len(items) < 5:
                continue
            name = items[0].strip()
            code = int(items[1])
            mnemonic = items[2].strip()
            latitude = float(items[3])
            longitude = float(items[4])
            altitude = float(items[5])
            category = "unknown"
            if len(items) > 6:
                category = items[6].strip()
            result.append(
                MadrigalInstrument(
                    name, code, mnemonic, latitude, longitude, altitude, category
                )
            )

        return result

    def getExperiments(
        self,
        experiment_codes: list[int],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        local: int = 1,
    ):
        """returns a list of all MadrigalExperiments that meet criteria at the given Madrigal site

        Inputs:

            code - int or list of ints representing instrument code(s). Special value of 0 selects all instruments.
            dt_start - datetime.datetime object representing start time of experiments desired
            dt_end - datetime.datetime object representing end time of experiments desired
            local - 0 if all sites desired, 1 (default) if only local experiments desired
        
        Outputs:

            List of MadrigalExperiment objects that meet the criteria.  Note that if the returned
            MadrigalExperiment is not local, the experiment id will be -1.  This means that you
            will need to create a new MadrigalData object with the url of the
            non-local experiment (MadrigalExperiment.madrigalUrl), and then call
            getExperiments a second time using that Madrigal url.  This is because
            while Madrigal sites share metadata about experiments, the real experiment ids are only
            known by the individual Madrigal sites.  See examples/exampleMadrigalWebServices.py
            for an example of this.
        """

        script_name = "getExperimentsService.py"

        url = self.cgiurl + script_name + "?"

        for exp_code in experiment_codes:
            url += f"code={exp_code}&"
        url += f"startyear={dt_start.year}&"
        url += f"startmonth={dt_start.month}&"
        url += f"startday={dt_start.day}&"
        url += f"starthour={dt_start.hour}&"
        url += f"startmin={dt_start.minute}&"
        url += f"startsec={dt_start.second}&"
        url += f"endyear={dt_end.year}&"
        url += f"endmonth={dt_end.month}&"
        url += f"endday={dt_end.day}&"
        url += f"endhour={dt_end.hour}&"
        url += f"endmin={dt_end.minute}&"
        url += f"endsec={dt_end.second}&"
        url += f"local={local}"

        # read main url
        response = requests.get(url, timeout=TIMEOUT2)
        if response.status_code != 200:
            raise ValueError("unable to open url " + str(url))
        page = response.text

        # parse the result
        if len(page) == 0:
            return []

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        for line in page:
            items = line.split(",")
            # calculate isLocal
            if len(items) < 20:
                continue
            if int(items[3]) == self.site_id:
                is_local = True
            else:
                is_local = False
            if is_local:
                experiment_id_str = items[0]
            else:
                experiment_id_str = "-1"
            if len(items) > 21:
                pi = items[20]
                pi_email = items[21]
            else:
                pi = "unknown"
                pi_email = "unknown"
            if len(items) > 23:
                uttimestamp = int(items[22])
                access = int(items[23])
            else:
                uttimestamp = None
                access = None

            result.append(
                MadrigalExperiment(
                    experiment_id_str,
                    items[1],
                    items[2],
                    items[3],
                    items[4],
                    items[5],
                    items[6],
                    items[7],
                    items[8],
                    items[9],
                    items[10],
                    items[11],
                    items[12],
                    items[13],
                    items[14],
                    items[15],
                    items[16],
                    items[17],
                    items[18],
                    is_local,
                    self.site_dict[int(items[3])],
                    pi,
                    pi_email,
                    uttimestamp,
                    access,
                    self._madrigal_version,
                )
            )

        return result

    def getExperimentFiles(self, id, getNonDefault=False):
        """returns a list of all default MadrigalExperimentFiles for a given experiment id

        Inputs:

           id - Experiment id.

           getNonDefault - if False (the default), only get default files, or realtime
                           files if no default files found.  If True, get all files.
                           In general, users should set this to False because default files
                           are the most reliable.

        Outputs:

            List of MadrigalExperimentFile objects for that experiment id


        """

        scriptName = "getExperimentFilesService.py"

        if int(id) == -1:
            err_str = """Illegal experiment id -1.  This is usually caused by calling
            getExperiments with the isLocal flag set to 0.  To get the experiment id for a non-local
            experiment, you will need to create a new MadrigalData object with the url of the 
            non-local experiment (MadrigalExperiment.madrigalUrl), and then call 
            getExperiments a second time using that Madrigal url.  This is because 
            while Madrigal sites share metadata about experiments, the real experiment ids are only
            known by the individual Madrigal sites. See examples/exampleMadrigalWebServices.py
            for an example of this.
            """
            raise ValueError(err_str)

        url = self.cgiurl + scriptName + "?id=%i" % (int(id))

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            return []

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # find out if no default files.  If so, return realtime also
        hasDefault = False
        for line in page:
            items = line.split(",")
            if len(items) < 4:
                continue
            if int(items[3]) == 1:
                hasDefault = True
                break

        for line in page:
            items = line.split(",")
            if len(items) < 4:
                continue
            category = int(items[3])
            if hasDefault and category != 1 and not getNonDefault:
                continue
            if not hasDefault and category != 4 and not getNonDefault:
                continue
            if len(items) > 6:
                doi = items[6]
            else:
                doi = None
            result.append(
                MadrigalExperimentFile(
                    items[0], items[1], items[2], items[3], items[4], items[5], id, doi
                )
            )

        return result

    def getExperimentFileParameters(self, fullFilename):
        """getExperimentFileParameters returns a list of all measured and derivable parameters in file

        Inputs:

           fullFilename - full path to experiment file as returned by getExperimentFiles.

        Outputs:

            List of MadrigalParameter objects for that fullFilename.  Includes both measured
            and derivable parameters in file.


        """

        scriptName = "getParametersService.py"

        url = self.cgiurl + scriptName + "?filename=%s" % (str(fullFilename))

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            return []

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        for line in page:
            items = line.split("\\")
            # with Madrigal 2.5, isAddIncrement was added as 8th column
            if len(items) < 7:
                continue
            try:
                isAddIncrement = int(items[7])
            except:
                isAddIncrement = -1
            result.append(
                MadrigalParameter(
                    items[0],
                    items[1],
                    int(items[2]),
                    items[3],
                    int(items[4]),
                    items[5],
                    int(items[6]),
                    isAddIncrement,
                )
            )

        return result

    def simplePrint(self, filename, user_fullname, user_email, user_affiliation):
        """simplePrint prints the data in the given file is a simple ascii format.

        simplePrint prints only the parameters in the file, without filters or derived
        parameters.  To choose which parameters to print, to print derived parameters, or
        to filter the data, use isprint instead.

        Inputs:

            filename - The absolute filename to be printed.  Returned by getExperimentFiles.

            user_fullname - full name of user making request

            user_email - email address of user making request

            user_affiliation - affiliation of user making request

        Returns: string representing all data in the file in ascii, space-delimited form.
                 The first line if the list of parameters printed.  The first six parameters will
                 always be year, month, day, hour, min, sec, representing the middle time of
                 the measurment.
        """
        parms = self.getExperimentFileParameters(filename)

        parmStr = "year,month,day,hour,min,sec"
        labelStr = "YEAR     MONTH       DAY      HOUR       MIN       SEC        "

        for parm in parms:
            if parm.isMeasured and parm.isAddIncrement != 1:
                parmStr += ",%s" % (parm.mnemonic)
                thisLabel = parm.mnemonic[:11].upper()
                labelStr += "%s%s" % (thisLabel, " " * (11 - len(thisLabel)))

        retStr = "%s\n" % (labelStr)

        retStr += self.isprint(
            filename, parmStr, "", user_fullname, user_email, user_affiliation
        )

        return retStr

    def isprint(
        self,
        filename,
        parms,
        filters,
        user_fullname,
        user_email,
        user_affiliation,
        outputFile=None,
    ):
        """returns as a string the isprint output given filename, parms, filters without headers or summary.

        Inputs:

            filename - The absolute filename to be analyzed by isprint.

            parms - Comma delimited string listing requested parameters (no spaces allowed).

            filters - Space delimited string listing filters desired, as in isprint command

            user_fullname - full name of user making request

            user_email - email address of user making request

            user_affiliation - affiliation of user making request

            outputFile - if not None, download the results to outputFile.  If outputFile has an extension
                of .h5, .hdf, or .hdf5, will download in Madrigal Hdf5 format.  If it has a .nc extension, will
                download as netCDF4. Otherwise, it will download as column delimited ascii.
                Trying to save as Hdf5 or netCDF4 with a Madrigal 2 site will raise an exception

        Returns:

            a string holding the isprint output, or None if writing to a file
        """

        scriptName = "isprintService.py"

        # build the complete cgi string, replacing characters as required by cgi standard

        url = self.cgiurl + scriptName + "?"

        url += "file=%s&" % (filename.replace("/", "%2F"))
        if parms.find(" ") != -1:
            parms = parms.replace(" ", "")  # remove all spaces
        parms = parms.replace("+", "%2B")
        parms = parms.split(",")
        for p in parms:
            url += "parms=%s&" % (p)
        filters = filters.replace("=", "%3D")
        filters = re.sub(",([a-zA-Z])", "?\g<1>", filters)
        filters = filters.replace(",", "%2C")
        filters = filters.replace("/", "%2F")
        filters = filters.replace("+", "%2B")
        filters = filters.replace(" ", "%20")
        url += "filters=%s&" % (filters)
        user_fullname = user_fullname.replace(" ", "+").strip()
        url += "user_fullname=%s&" % (user_fullname)
        user_email = user_email.strip()
        url += "user_email=%s&" % (user_email)
        user_affiliation = user_affiliation.replace(" ", "+").strip()
        url += "user_affiliation=%s" % (user_affiliation)
        if not outputFile is None:
            url += "&output=%s" % (os.path.basename(outputFile))
            filename, file_extension = os.path.splitext(outputFile)
            if file_extension in (".hdf5", ".h5", ".hdf"):
                format = "Hdf5"
            elif file_extension in (".nc",):
                format = "netCDF4"
            else:
                format = "ascii"
            # if Hdf5 or netCDF4, make sure site is 3 or greater
            if format in ("Hdf5", "netCDF4"):
                version = self.getVersion()
                if packaging.version.parse(version) < packaging.version.parse("3.0"):
                    raise ValueError(
                        "Madrigal site at %s is below 3.0, cannot convert to Hdf5 or netCDF4"
                        % (self.cgiurl)
                    )
        else:
            format = "ascii"

        # read main url
        url = url.replace("+", "%2B")
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT)
        except:
            raise ValueError("unable to open url " + str(url))

        if format == "ascii":
            page = mainUrl.read().decode("utf-8")
        else:
            page = mainUrl.read()

        mainUrl.close()

        if format == "ascii":
            if page.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url))

        if outputFile is None:
            return page

        else:
            if format == "ascii":
                f = open(outputFile, "w")
            else:
                f = open(outputFile, "wb")
            f.write(page)
            f.close()

    def madCalculator(
        self,
        year,
        month,
        day,
        hour,
        min,
        sec,
        startLat,
        endLat,
        stepLat,
        startLong,
        endLong,
        stepLong,
        startAlt,
        endAlt,
        stepAlt,
        parms,
        oneDParmList=[],
        oneDParmValues=[],
    ):
        """

        Input arguments:

            1. year - int

            2. month - int

            3. day - int

            4. hour - int

            5. min - int

            6. sec - int

            7. startLat - Starting geodetic latitude, -90 to 90 (float)

            8. endLat - Ending geodetic latitude, -90 to 90 (float)

            9. stepLat - Latitude step (0.1 to 90) (float)

            10. startLong - Starting geodetic longitude, -180 to 180  (float)

            11. endLong - Ending geodetic longitude, -180 to 180 (float)

            12. stepLong - Longitude step (0.1 to 180) (float)

            13. startAlt - Starting geodetic altitude, >= 0 (float)

            14. endAlt - Ending geodetic altitude, > 0 (float)

            15. stepAlt - Altitude step (>= 0.1) (float)

            16. parms - comma delimited string of Madrigal parameters desired

            17. oneDParmList - a list of one-D parameters whose values should
                               be set for the calculation.  Can be codes or mnemonics.
                               Defaults to empty list.

            18. oneDParmValues - a list of values (doubles) associated with the one-D
                                 parameters specified in oneDParmList. Defaults to empty list.

        Returns:

            A list of lists of doubles, where each list contains 3 + number of parameters doubles.
            The first three doubles are the input latitude, longitude, and altitude.  The rest of the
            doubles are the values of each of the calculated values.  If the value cannot be calculated,
            it will be set to nan.

            Example:

                result = testData.madCalculator(1999,2,15,12,30,0,45,55,5,-170,-150,10,200,200,0,'bmag,bn')

                result = [  [45.0, -170.0, 200.0, 4.1315700000000002e-05, 2.1013500000000001e-05]
                            [45.0, -160.0, 200.0, 4.2336899999999998e-05, 2.03685e-05]
                            [45.0, -150.0, 200.0, 4.3856400000000002e-05, 1.97411e-05]
                            [50.0, -170.0, 200.0, 4.3913599999999999e-05, 1.9639999999999998e-05]
                            [50.0, -160.0, 200.0, 4.4890099999999999e-05, 1.8870999999999999e-05]
                            [50.0, -150.0, 200.0, 4.6337800000000002e-05, 1.80077e-05]
                            [55.0, -170.0, 200.0, 4.6397899999999998e-05, 1.78115e-05]
                            [55.0, -160.0, 200.0, 4.7265400000000003e-05, 1.6932500000000001e-05]
                            [55.0, -150.0, 200.0, 4.85495e-05,            1.5865399999999999e-05] ]

                Columns:     gdlat  glon    gdalt  bmag                    bn

        """

        scriptName = "madCalculatorService.py"

        url = self.cgiurl + scriptName + "?year"

        if len(oneDParmList) != len(oneDParmValues):
            raise ValueError("len(oneDParmList) != len(oneDParmValues)")

        if parms.find(" ") != -1:
            parms = parms.replace(" ", "")  # remove all spaces

        # append arguments
        url += "=%i&month" % (int(year))
        url += "=%i&day" % (int(month))
        url += "=%i&hour" % (int(day))
        url += "=%i&min" % (int(hour))
        url += "=%i&sec" % (int(min))
        url += "=%i&startLat" % (int(sec))
        url += "=%f&endLat" % (float(startLat))
        url += "=%f&stepLat" % (float(endLat))
        url += "=%f&startLong" % (float(stepLat))
        url += "=%f&endLong" % (float(startLong))
        url += "=%f&stepLong" % (float(endLong))
        url += "=%f&startAlt" % (float(stepLong))
        url += "=%f&endAlt" % (float(startAlt))
        url += "=%f&stepAlt" % (float(endAlt))
        url += "=%f&parms" % (float(stepAlt))
        url += "=%s" % (parms)

        for i in range(len(oneDParmList)):
            url += "&oneD=%s,%s" % (str(oneDParmList[i]), str(oneDParmValues[i]))

        # remove any pluses in the url due to scientific notation
        url = url.replace("+", "%2B")

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # parse output
        for line in page:
            items = line.split()
            if len(items) < 3:
                # blank line
                continue
            newList = []
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def madCalculator2(
        self,
        year,
        month,
        day,
        hour,
        min,
        sec,
        latList,
        lonList,
        altList,
        parms,
        oneDParmList=[],
        oneDParmValues=[],
        twoDParmList=[],
        twoDParmValues=[],
    ):
        """
        madCalculator2 is similar to madCalculator, except that a random collection of points in space can be specified,
        rather than a grid of points.  Also, a user can input 2D data.

        Added to Madrigal2.6 as web service - will not run on earlier Madrigal installations.

        Input arguments:

            1. year - int

            2. month - int

            3. day - int

            4. hour - int

            5. min - int

            6. sec - int

            7. latList - a list of geodetic latitudes, -90 to 90

            8. lonList - a list of longitudes, -180 to 180. Length must = lats

            9. altList - a list of geodetic altitudes in km. Length must = lats

            10. parms - comma delimited string of Madrigal parameters desired

            11. oneDParmList - a list of one-D parameters whose values should
                               be set for the calculation.  Can be codes or mnemonics.
                               Defaults to empty list.

            12. oneDParmValues - a list of values (doubles) associated with the one-D
                                 parameters specified in oneDParmList. Defaults to empty list.

            13. twoDParmList - a python list of two-D parameters as mnemonics.  Defaults to [].

            14. twoDParmValues - a python list of lists of len = len(twoDParmList). Each individual
                             list is a list of doubles representing values of the two-D
                             parameter set in twoDParmList, with a length = number
                             of points (or equal to len(lats)). Defaults to [].

        Returns:

            A list of lists of doubles, where each list contains 3 + number of parameters doubles.
            The first three doubles are the input latitude, longitude, and altitude.  The rest of the
            doubles are the values of each of the calculated values.  If the value cannot be calculated,
            it will be set to nan.

            Example:

                result = testData.madCalculator2(1999,2,15,12,30,0,[45,55],[-170,-150],[200,300],'sdwht,kp')

                result = [ [1999.0, 2.0, 15.0, 12.0, 30.0, 0.0, 3.0, 15.0]
                           [1999.0, 2.0, 15.0, 12.0, 45.0, 0.0, 3.0, 15.0]
                           [1999.0, 2.0, 15.0, 13.0, 0.0, 0.0, 3.0, 15.0]
                           [1999.0, 2.0, 15.0, 13.0, 15.0, 0.0, 3.0, 15.0] ]


                Columns:     gdlat  glon    gdalt  sdwht   kp

        Now uses POST to avoid long url issue

        """

        # verify Madrigal site can call this command
        if self.compareVersions("2.6", self._madrigal_version):
            raise IOError(
                "madCalculator2 requires Madrigal 2.6 or greater, but this site is version %s"
                % (self._madrigal_version)
            )

        scriptName = "madCalculator2Service.py"

        if parms.find(" ") != -1:
            parms = parms.replace(" ", "")  # remove all spaces

        url = self.cgiurl + scriptName

        # error checking
        if len(oneDParmList) != len(oneDParmValues):
            raise ValueError("len(oneDParmList) != len(oneDParmValues)")

        if len(latList) == 0:
            raise ValueError("length of latList must be at least one")

        if len(latList) != len(lonList) or len(latList) != len(altList):
            raise ValueError("lengths of latList, lonList, altList must all be equal")

        if len(oneDParmList) != len(oneDParmValues):
            raise ValueError("len(oneDParmList) != len(oneDParmValues)")

        postUrl = "year"
        # append arguments
        delimiter = ","
        postUrl += "=%i&month" % (int(year))
        postUrl += "=%i&day" % (int(month))
        postUrl += "=%i&hour" % (int(day))
        postUrl += "=%i&min" % (int(hour))
        postUrl += "=%i&sec" % (int(min))
        postUrl += "=%i&lats=" % (int(sec))
        for i in range(len(latList)):
            postUrl += str(latList[i])
            if i + 1 < len(latList):
                postUrl += ","
        postUrl += "&longs="
        for i in range(len(lonList)):
            postUrl += str(lonList[i])
            if i + 1 < len(lonList):
                postUrl += ","
        postUrl += "&alts="
        for i in range(len(altList)):
            postUrl += str(altList[i])
            if i + 1 < len(altList):
                postUrl += ","
        postUrl += "&parms=%s" % (parms)

        for i in range(len(oneDParmList)):
            postUrl += "&oneD=%s,%s" % (str(oneDParmList[i]), str(oneDParmValues[i]))

        for i in range(len(twoDParmList)):
            postUrl += "&twoD=%s," % (str(twoDParmList[i]))
            for j in range(len(twoDParmValues[i])):
                postUrl += str(twoDParmValues[i][j])
                if j + 1 < len(twoDParmValues[i]):
                    postUrl += ","

        # remove any pluses in the url due to scientific notation
        postUrl = postUrl.replace("+", "%2B")

        data = postUrl.encode("utf-8")

        # read main url
        try:
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, data=data, timeout=TIMEOUT)
        except:
            raise ValueError("unable to open url " + str((url, postUrl)))

        page = response.read().decode("utf8")
        page = page.split("\n")

        response.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # parse output
        for line in page:
            items = line.split()
            if len(items) < 3:
                # blank line
                continue
            newList = []
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def madCalculator3(
        self,
        yearList,
        monthList,
        dayList,
        hourList,
        minList,
        secList,
        latList,
        lonList,
        altList,
        parms,
        oneDParmList=[],
        oneDParmValues=[],
        twoDParmList=[],
        twoDParmValues=[],
    ):
        """
        madCalculator3 is similar to madCalculator, except that multiple times can be specified,
        where each time can have its own unique spatial positions and 1D and 2D parms. It is
        equivalent to multiple calls to madCalculator2, except that it should greatly improve
        performance where multiple calls to madCalculator2 are required for different times.
        The only restriction is that the same parameters must be requested for every time.

        Added to Madrigal2.6 as web service - will not run on earlier Madrigal installations.

        Now uses POST to send arguments, due to large volume of data possible

        Input arguments:

            1. yearList - a list of years, one for each time requested (ints)

            2. monthList - a list of months, one for each time requested. (ints)

            3. dayList - a list of days, one for each time requested. (ints)

            4. hourList - a list of hours, one for each time requested. (ints)

            5. minList - a list of minutes, one for each time requested. (ints)

            6. secList - a list of seconds, one for each time requested. (ints)

            7. latList - a list of lists of geodetic latitudes, -90 to 90.  The
               first list is for the first time, the second for the second time, etc.
               The list do not need to have the same number of points.  The number
               of times must match yearList.
               Data organization: latList[timeIndex][positionIndex]

            8. lonList - a list of lists of longitudes, -180 to 180. The
               first list is for the first time, the second for the second time, etc.
               The list do not need to have the same number of points.  The number
               of times must match yearList. Lens must match latList
               Data organization: lonList[timeIndex][positionIndex]

            9. altList - a list of lists of geodetic altitudes in km. The
               first list is for the first time, the second for the second time, etc.
               The list do not need to have the same number of points.  The number
               of times must match yearList. Lens must match latList
               Data organization: altList[timeIndex][positionIndex]

            10. parms - comma delimited string of Madrigal parameters desired

            11. oneDParmList - a list of one-D parameters whose values should
                               be set for the calculation.  Can be codes or mnemonics.
                               Defaults to empty list.

            12. oneDParmValues - a list of lists of values (doubles) associated with the one-D
                                 parameters specified in oneDParmList. Defaults to empty list.
                                 The first list is for the first 1D parameter in oneDParmList,
                                 and must be on length len(yearList).  The second list is for
                                 the second parameter, etc.
                                 Data organization: onDParmValues[parameterIndex][timeIndex]

            13. twoDParmList - a python list of of two-D parameters as mnemonics.  Defaults to [].

            14. twoDParmValues - a list of lists of lists of values (doubles) associated with the two-D
                                 parameters specified in twoDParmList. Defaults to empty list.
                                 The first list is for the first 2D parameter in oneDParmList,
                                 and must be a list of length len(yearList).  Each list in that
                                 list must be of len(num positions for that time). The second list is for
                                 the second parameter, etc.
                                 Data organization: twoDParmValues[parameterIndex][timeIndex][positionIndex]

        Returns:

            A list of lists of doubles, where each list contains 9 + number of parameters doubles.
            The first nine doubles are:
            1) year, 2) month, 3) day, 4) hour, 5) minute, 6) second,
            7) input latitude, 8) longitude, and 9) altitude.
            The rest of the doubles are the values of each of the calculated values.  If the value
            cannot be calculated, it will be set to nan.

            Example:

                testData.madCalculator3(yearList=[2001,2001], monthList=[3,3], dayList=[19,20],
                                     hourList=[12,12], minList=[30,40], secList=[20,0],
                                     latList=[[45,46,47,48.5],[46,47,48.2,49,50]],
                                     lonList=[[-70,-71,-72,-73],[-70,-71,-72,-73,-74]],
                                     altList=[[145,200,250,300.5],[200,250,300,350,400]],
                                     parms='bmag,pdcon,ne_model',
                                     oneDParmList=['kinst','elm'],
                                     oneDParmValues=[[31.0,31.0],[45.0,50.0]],
                                     twoDParmList=['ti','te','ne'],
                                     twoDParmValues=[[[1000,1000,1000,1000],[1000,1000,1000,1000,1000]],
                                                     [[1100,1200,1300,1400],[1500,1000,1100,1200,1300]],
                                                     [[1.0e10,1.0e10,1.0e10,1.0e10],[1.0e10,1.0e10,1.0e10,1.0e10,1.0e10]]])


                Columns:     year month day hour minute second gdlat  glon  gdalt  bmag  pdcon  ne_model

        """
        # verify Madrigal site can call this command
        if self.compareVersions("2.6", self._madrigal_version):
            raise IOError(
                "madCalculator3 requires Madrigal 2.6 or greater, but this site is version %s"
                % (self._madrigal_version)
            )

        scriptName = "madCalculator3Service.py"

        url = self.cgiurl + scriptName

        postUrl = ""

        # error checking
        try:
            totalTimes = len(yearList)
        except:
            raise ValueError("yearList must be a list, not %s" % (str(yearList)))

        if (
            len(monthList) != totalTimes
            or len(dayList) != totalTimes
            or len(hourList) != totalTimes
            or len(minList) != totalTimes
            or len(secList) != totalTimes
        ):
            raise ValueError("Not all time lists have same length")

        if parms.find(" ") != -1:
            parms = parms.replace(" ", "")  # remove all spaces

        # add time arguments
        postUrl += "year="
        for i, year in enumerate(yearList):
            postUrl += "%i" % (year)
            if i + 1 < len(yearList):
                postUrl += ","

        postUrl += "&month="
        for i, month in enumerate(monthList):
            postUrl += "%i" % (month)
            if i + 1 < len(monthList):
                postUrl += ","

        postUrl += "&day="
        for i, day in enumerate(dayList):
            postUrl += "%i" % (day)
            if i + 1 < len(dayList):
                postUrl += ","

        postUrl += "&hour="
        for i, hour in enumerate(hourList):
            postUrl += "%i" % (hour)
            if i + 1 < len(hourList):
                postUrl += ","

        postUrl += "&min="
        for i, minute in enumerate(minList):
            postUrl += "%i" % (minute)
            if i + 1 < len(minList):
                postUrl += ","

        postUrl += "&sec="
        for i, sec in enumerate(secList):
            postUrl += "%i" % (sec)
            if i + 1 < len(secList):
                postUrl += ","

        # get numPos list from latList
        numPos = []
        for lats in latList:
            numPos.append(len(lats))

        postUrl += "&numPos="
        for i, pos in enumerate(numPos):
            postUrl += "%i" % (pos)
            if i + 1 < len(numPos):
                postUrl += ","

        postUrl += "&lats="
        for i, posList in enumerate(latList):
            if len(posList) != numPos[i]:
                raise ValueError("mismatched number of points in latList")
            for j, pos in enumerate(posList):
                postUrl += "%f" % (pos)
                if i + 1 < len(latList) or j + 1 < len(posList):
                    postUrl += ","

        postUrl += "&longs="
        for i, posList in enumerate(lonList):
            if len(posList) != numPos[i]:
                raise ValueError("mismatched number of points in lonList")
            for j, pos in enumerate(posList):
                postUrl += "%f" % (pos)
                if i + 1 < len(lonList) or j + 1 < len(posList):
                    postUrl += ","

        postUrl += "&alts="
        for i, posList in enumerate(altList):
            if len(posList) != numPos[i]:
                raise ValueError("mismatched number of points in altList")
            for j, pos in enumerate(posList):
                postUrl += "%f" % (pos)
                if i + 1 < len(altList) or j + 1 < len(posList):
                    postUrl += ","

        postUrl += "&parms=%s" % (parms)

        if len(oneDParmList) != len(oneDParmValues):
            raise ValueError("len(oneDParmList) != len(oneDParmValues)")

        for i, parm in enumerate(oneDParmList):
            postUrl += "&oneD=%s," % (str(parm))
            if len(oneDParmValues[i]) != totalTimes:
                raise ValueError("wrong number of 1D parms for %s" % (str(parm)))
            for j, value in enumerate(oneDParmValues[i]):
                postUrl += "%f" % (value)
                if j + 1 < len(oneDParmValues[i]):
                    postUrl += ","

        for i, parm in enumerate(twoDParmList):
            postUrl += "&twoD=%s," % (str(parm))
            if len(twoDParmValues[i]) != totalTimes:
                raise ValueError("wrong number of 2D parms for %s" % (str(parm)))
            for j, valueList in enumerate(twoDParmValues[i]):
                if len(valueList) != numPos[j]:
                    raise ValueError("wrong number of 2D parms for %s" % (str(parm)))
                for k, value in enumerate(valueList):
                    postUrl += "%f" % (value)
                    if j + 1 < len(twoDParmValues[i]) or k + 1 < len(valueList):
                        postUrl += ","

        # remove any pluses in the url due to scientific notation
        postUrl = postUrl.replace("+", "%2B")

        data = postUrl.encode("utf8")

        # read main url
        try:
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, data=data, timeout=TIMEOUT)
        except:
            raise ValueError("unable to open url " + str((url, postUrl)))

        page = response.read().decode("utf8")
        page = page.split("\n")

        response.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # time data to add to each line
        year = None
        month = None
        day = None
        hour = None
        minute = None
        second = None

        for line in page:
            items = line.split()
            if len(items) < 3:
                # blank line
                continue
            if line.find("TIME") != -1:
                # new time found
                dates = items[1].split("/")
                year = int(dates[2])
                month = int(dates[0])
                day = int(dates[1])
                times = items[2].split(":")
                hour = int(times[0])
                minute = int(times[1])
                second = int(times[2])
                continue
            newList = [year, month, day, hour, minute, second]
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def madTimeCalculator(
        self,
        startyear,
        startmonth,
        startday,
        starthour,
        startmin,
        startsec,
        endyear,
        endmonth,
        endday,
        endhour,
        endmin,
        endsec,
        stephours,
        parms,
    ):
        """

        Input arguments:

            1. startyear - int

            2. startmonth - int

            3. startday - int

            4. starthour - int

            5. startmin - int

            6. startsec - int

            7. endyear - int

            8. endmonth - int

            9. endday - int

            10. endhour - int

            11. endmin - int

            12. endsec - int

            13. stephours - float - number of hours per time step

            14. parms - comma delimited string of Madrigal parameters desired (must not depend on location)

        Returns:

            A list of lists, where each list contains 6 ints (year, month, day, hour, min, sec)  + number
            of parameters.  If the value cannot be calculated, it will be set to nan.

            Example:

                result = testData.madTestCalculator(1999,2,15,12,30,0,
                                                    1999,2,20,12,30,0,
                                                    24.0, 'kp,dst')

                result = [[1999.0, 2.0, 15.0, 12.0, 30.0, 0.0, 3.0, -9.0]
                          [1999.0, 2.0, 16.0, 12.0, 30.0, 0.0, 1.0, -6.0]
                          [1999.0, 2.0, 17.0, 12.0, 30.0, 0.0, 4.0, -31.0]
                          [1999.0, 2.0, 18.0, 12.0, 30.0, 0.0, 6.7000000000000002, -93.0]
                          [1999.0, 2.0, 19.0, 12.0, 30.0, 0.0, 5.2999999999999998, -75.0]]

                Columns:     year, month, day, hour, min, sec, kp, dst

        """

        scriptName = "madTimeCalculatorService.py"

        url = self.cgiurl + scriptName + "?startyear"

        if parms.find(" ") != -1:
            parms = parms.replace(" ", "")  # remove all spaces

        # append arguments
        url += "=%i&startmonth" % (int(startyear))
        url += "=%i&startday" % (int(startmonth))
        url += "=%i&starthour" % (int(startday))
        url += "=%i&startmin" % (int(starthour))
        url += "=%i&startsec" % (int(startmin))
        url += "=%i&endyear" % (int(startsec))
        url += "=%i&endmonth" % (int(endyear))
        url += "=%i&endday" % (int(endmonth))
        url += "=%i&endhour" % (int(endday))
        url += "=%i&endmin" % (int(endhour))
        url += "=%i&endsec" % (int(endmin))
        url += "=%i&stephours" % (int(endsec))
        url += "=%f&parms" % (float(stephours))
        url += "=%s" % (parms)

        # remove any pluses in the url due to scientific notation
        url = url.replace("+", "%2B")

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # parse output
        for line in page:
            items = line.split()
            if len(items) < 3:
                # blank line
                continue
            newList = []
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def radarToGeodetic(self, slatgd, slon, saltgd, az, el, radarRange):
        """radarToGeodetic converts arrays of az, el, and ranges to geodetic locations.

        Input arguments:

            1. slatgd - radar geodetic latitude

            2. slon - radar longitude

            3. saltgd - radar altitude

            4. az - either a single azimuth, or a list of azimuths

            5. el - either a single elevation, or a list of elevations.  If so, len(el)
                    must = len(az)

            6. radarRange - either a single range, or a list of ranges.  If so, len(radarRange)
                            must = len(az)


        Returns:

            A list of lists, where each list contains 3 floats (gdlat, glon, and gdalt)
        """
        scriptName = "radarToGeodeticService.py"

        url = self.cgiurl + scriptName + "?slatgd"

        # append arguments
        url += "=%f&slon" % (float(slatgd))
        url += "=%f&saltgd" % (float(slon))
        url += "=%f&" % (float(saltgd))

        if type(az) == list or type(az) == tuple:
            if len(az) != len(el) or len(az) != len(radarRange):
                raise ValueError("all lists most have same length")
            for i in range(len(az)):
                if i == 0:
                    arg = str(az[i])
                else:
                    arg += "," + str(az[i])
            url += "az=%s&" % (arg)

            for i in range(len(el)):
                if i == 0:
                    arg = str(el[i])
                else:
                    arg += "," + str(el[i])
            url += "el=%s&" % (arg)

            for i in range(len(radarRange)):
                if i == 0:
                    arg = str(radarRange[i])
                else:
                    arg += "," + str(radarRange[i])
            url += "range=%s" % (arg)

        else:
            url += "az=%f&" % (az)
            url += "el=%f&" % (el)
            url += "range=%f&" % (radarRange)

        # remove any pluses in the url due to scientific notation
        url = url.replace("+", "%2B")

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # parse output
        for line in page:
            items = line.split(",")
            if len(items) < 3:
                # blank line
                continue
            newList = []
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def geodeticToRadar(self, slatgd, slon, saltgd, gdlat, glon, gdalt):
        """geodeticToRadar converts arrays of points in space to az, el, and range.

        Input arguments:

            1. slatgd - radar geodetic latitude

            2. slon - radar longitude

            3. saltgd - radar altitude

            4. gdlat - either a single geodetic latitude, or a list of geodetic latitudes

            5. glon - either a single longitude, or a list of longitudes.  If so, len(gdlat)
                      must = len(glon)

            6. gdalt - either a single deodetic altitude, or a list of geodetic altitudes.
                       If so, len(gdalt) must = len(gdlat)


        Returns:

            A list of lists, where each list contains 3 floats (az, el, and range)
        """
        scriptName = "geodeticToRadarService.py"

        url = self.cgiurl + scriptName + "?slatgd"

        # append arguments
        url += "=%f&slon" % (float(slatgd))
        url += "=%f&saltgd" % (float(slon))
        url += "=%f&" % (float(saltgd))

        if type(gdlat) == list or type(gdlat) == tuple:
            if len(gdlat) != len(glon) or len(gdlat) != len(gdalt):
                raise ValueError("all lists most have same length")
            for i in range(len(gdlat)):
                if i == 0:
                    arg = str(gdlat[i])
                else:
                    arg += "," + str(gdlat[i])
            url += "gdlat=%s&" % (arg)

            for i in range(len(glon)):
                if i == 0:
                    arg = str(glon[i])
                else:
                    arg += "," + str(glon[i])
            url += "glon=%s&" % (arg)

            for i in range(len(gdalt)):
                if i == 0:
                    arg = str(gdalt[i])
                else:
                    arg += "," + str(gdalt[i])
            url += "gdalt=%s" % (arg)

        else:
            url += "gdlat=%f&" % (gdlat)
            url += "glon=%f&" % (glon)
            url += "gdalt=%f&" % (gdalt)

        # remove any pluses in the url due to scientific notation
        url = url.replace("+", "%2B")

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that error was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        # parse output
        for line in page:
            items = line.split(",")
            if len(items) < 3:
                # blank line
                continue
            newList = []
            for item in items:
                try:
                    newList.append(float(item))
                except:
                    newList.append(str(item))
            result.append(newList)

        return result

    def downloadFile(
        self,
        filename,
        destination,
        user_fullname,
        user_email,
        user_affiliation,
        format="simple",
    ):
        """downloadFile will download a Cedar file in the specified format.

        Inputs:

            filename - The absolute filename to as returned via getExperimentFiles.

            destination - where the file is to be stored

            user_fullname - full name of user making request

            user_email - email address of user making request

            user_affiliation - affiliation of user making request

            format - file format desired.  May be  'simple', 'hdf5', 'netCDF4',
                'madrigal', 'blockedBinary', 'ncar',
                 'unblockedBinary', or 'ascii'.  Default is 'simple'
                 Simple is a simple ascii space delimited column format.
                 Simple and hdf5 are recommended since they are standard formats

                hdf5 format works for Madrigal 2.6 or later
                netCDF4 format works for Madrigal 3.0 or later
                madrigal', 'blockedBinary', 'ncar',
                 'unblockedBinary', or 'ascii' no longer supported for Madrigal 3.

        """
        fileType = 0
        if format not in (
            "hdf5",
            "madrigal",
            "blockedBinary",
            "ncar",
            "unblockedBinary",
            "ascii",
            "simple",
            "netCDF4",
        ):
            raise ValueError("Illegal format specified: %s" % (str(format)))
        if format == "blockedBinary":
            fileType = 1
        elif format == "ncar":
            fileType = 2
        elif format == "unblockedBinary":
            fileType = 3
        elif format == "simple":
            fileType = -1
        elif format == "hdf5":
            # verify Madrigal site can handle this argument
            if self.compareVersions("2.6", self._madrigal_version):
                raise IOError(
                    "downloadFile with hdf5 format requires Madrigal 2.6 or greater, but this site is version %s"
                    % (self._madrigal_version)
                )
            fileType = -2
        elif format == "netCDF4":
            # verify Madrigal site can handle this argument
            if self.compareVersions("3.0", self._madrigal_version):
                raise IOError(
                    "downloadFile with netCDF4 format requires Madrigal 3.0 or greater, but this site is version %s"
                    % (self._madrigal_version)
                )
            fileType = -3
        else:
            fileType = 4

        # verify no old formats specified if Madrigal 3
        if fileType > 0:
            if not self.compareVersions("3.0", self._madrigal_version):
                raise IOError(
                    "Only Madrigal 2.X sites can create old style files, but this site is version %s"
                    % (self._madrigal_version)
                )

        url = urllib.parse.urljoin(
            self.cgiurl,
            "getMadfile.cgi?fileName=%s&fileType=%i&" % (filename, fileType),
        )
        user_fullname = user_fullname.replace(" ", "+").strip()
        url += "user_fullname=%s&" % (user_fullname)
        user_email = user_email.strip()
        url += "user_email=%s&" % (user_email)
        user_affiliation = user_affiliation.replace(" ", "+").strip()
        url += "user_affiliation=%s" % (user_affiliation)
        url = url.replace("+", "%2B")

        CHUNK = 16 * 1024

        urlFile = urllib.request.urlopen(url, timeout=TIMEOUT)

        if format in ("ascii", "simple"):
            f = open(destination, "w")
        else:
            f = open(destination, "wb")

        while True:
            if format in ("ascii", "simple"):
                try:
                    data = urlFile.read(CHUNK).decode("utf8")
                except:
                    # probably gzip ascii - convert
                    f.close()
                    try:
                        os.remove(destination)
                    except:
                        pass
                    urlFile.close()
                    urlFile = urllib.request.urlopen(url, timeout=TIMEOUT)
                    f = open(destination + ".gz", "wb")
                    format = "gzip"
                    data = urlFile.read(CHUNK)

            else:
                data = urlFile.read(CHUNK)
            if not data:
                break
            f.write(data)

        urlFile.close()

        f.close()

    def listFileTimes(self, expDir=None):
        """listFileTimes lists the filenames and last modification times for all files in a Madrigal database.

        Inputs: expDir - experiment directory to which to start.  May be any directory or subdirectory below
            experiments[0-9]*.  Path may be absolute or relative to experiments[0-9]*.  If None (the default),
            include entire experiments[0-9]* directory(s).  Examples: ('/opt/madrigal/experiments/1998',
            'experiments/2002/gps')

        Returns: a list of tuple of 1. filename relative to experiments[0-9]* directory, and 2) datetime in UT of
            last file modification

        Requires:  Madrigal 2.6 or greater
        """
        # verify Madrigal site can call this command
        if self.compareVersions("2.6", self._madrigal_version):
            raise IOError(
                "listFileTimes requires Madrigal 2.6 or greater, but this site is version %s"
                % (self._madrigal_version)
            )

        url = urllib.parse.urljoin(self.cgiurl, "listFileTimesService.py")

        if expDir:
            url += "?expDir=%s" % (expDir)

        url = url.replace("+", "%2B")

        urlFile = urllib.request.urlopen(url, timeout=TIMEOUT)

        retList = []

        data = urlFile.read().decode("utf-8")
        lines = data.split("\n")
        for line in lines:
            items = line.split(",")
            if len(items) != 2:
                continue
            filename = items[0].strip()[1:-1]  # strip off quotes
            dt = datetime.datetime.strptime(items[1].strip(), "%Y-%m-%d %H:%M:%S")
            retList.append((filename, dt))

        return retList

    def downloadWebFile(self, expPath, destination):
        """downloadWebFile allows a user to download a axillary file from the web site. Used to
        download files found by listFileTimes.

        Requires a Madrigal 3.0 site.

        Inputs:
            expPath - filename relative to experiments[0-9]* directory. As returned by listFileTimes.
            destination - path to save file to
        """
        # verify Madrigal site can call this command
        if self.compareVersions("3.0", self._madrigal_version):
            raise IOError(
                "downloadWebFile requires Madrigal 3.0 or greater, but this site is version %s"
                % (self._madrigal_version)
            )

        url = urllib.parse.urljoin(self.cgiurl, "downloadWebFileService.py")

        url += "?expPath=%s" % (expPath.replace(" ", "+"))
        url = url.replace("+", "%2B")

        urlFile = urllib.request.urlopen(url, timeout=TIMEOUT2)

        data = urlFile.read()

        urlFile.close()

        f = open(destination, "wb")

        f.write(data)

        f.close()

    def traceMagneticField(
        self,
        year,
        month,
        day,
        hour,
        minute,
        second,
        inputType,
        outputType,
        alts,
        lats,
        lons,
        model,
        qualifier,
        stopAlt=None,
    ):
        """
        traceMagneticField returns a point along a magnetic field line for each
        point specified by the lists alts, lats, lons.
        Traces to either 1) conjugate point, 2) intersection with a given altitude in the
        northern or southern hemisphere, 3) to the apex, or 4) to GSM XY plane, depending on qualifier
        argument.  Uses Tsyganenko or IGRF fields, depending on model argument.
        Input arguments are either GSM or Geodetic, depending on inputType argument.
        Output arguments are either GSM or Geodetic, depending on outputType
        argument.

        Inputs:

            year, month, day, hour, minute, second - time at which to do the trace

            inputType  - 0 for geodetic, 1 for GSM

            outputType - 0 for geodetic, 1 for GSM

            The following parameter depend on inputType:

            alts - a list of geodetic altitudes or ZGSMs of starting point

            lats - a clist of geodetic latitudes or XGSMs of starting point

            lons - a list of longitude or YGSM of starting point

            Length of all three lists must be the same

            model - 0 for Tsyganenko, 1 for IGRF

            qualifier - 0 for conjugate, 1 for north_alt, 2 for south_alt, 3 for apex, 4 for GSM XY plane

            stopAlt - altitude in km to stop trace at, if qualifier is north_alt or south_alt.
                If other qualifier, this parameter is not required. Default is None, which will raise
                exception if qualifier is north_alt or south_alt

        Returns a tuple of tuples, one tuple for point in (alts, lats, lons) lists, where each tuple has
        three items:

            1. geodetic altitude or ZGSM of ending point

            2. geodetic latitude or XGSM of ending point

            3. longitude or YGSM of ending point


        If error, traceback includes error description
        """
        scriptName = "traceMagneticFieldService.py"

        url = self.cgiurl + scriptName + "?"

        delimiter = ","

        # append arguments
        url += "year=%i&" % (int(year))
        url += "month=%i&" % (int(month))
        url += "day=%i&" % (int(day))
        url += "hour=%i&" % (int(hour))
        url += "min=%i&" % (int(minute))
        url += "sec=%i&" % (int(second))
        url += "inputType=%i&" % (int(inputType))
        url += "outputType=%i&" % (int(outputType))
        in1Str = delimiter.join([str(item) for item in alts])
        url += "in1=%s&" % (in1Str)
        in2Str = delimiter.join([str(item) for item in lats])
        url += "in2=%s&" % (in2Str)
        in3Str = delimiter.join([str(item) for item in lons])
        url += "in3=%s&" % (in3Str)
        url += "model=%i&" % (int(model))
        url += "qualifier=%i&" % (int(qualifier))
        if stopAlt == None:
            if int(qualifier) in (1, 2):
                raise ValueError("stopAlt must be set for qualifer in (1,2)")
            else:
                stopAlt = 0.0
        url += "stopAlt=%s" % (str(stopAlt))

        # remove any pluses in the url due to scientific notation
        url = url.replace("+", "%2B")

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.split("\n")

        mainUrl.close()

        # parse the result
        if len(page) == 0:
            raise ValueError("No data found at url" + str(url))

        # check that html was not returned
        for line in page:
            if line.find("Error occurred") != -1:
                raise ValueError("error raised using url " + str(url) + " " + str(page))

        result = []

        for line in page:
            items = line.split(",")
            if len(items) < 3:
                continue
            result.append((float(items[0]), float(items[1]), float(items[2])))

        return result

    def getVersion(self):
        """getVersion gets the version number of Madrigal in form number dot number etc.

        Assumes version is 2.5 if no getVersionService.py installed
        """
        scriptName = "getVersionService.py"

        url = self.cgiurl + scriptName

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            # if this fails, must be 2.5
            return "2.5"

        page = mainUrl.read().decode("utf-8")

        mainUrl.close()

        return page.strip()

    def compareVersions(self, ver1, ver2):
        """compareVersions returns False if ver1 <= ver2, 0 True otherwise

        Inputs: version number strings, in form number dot number (any number of dots)
        """
        return packaging.version.parse(ver1) > packaging.version.parse(ver2)

    def getCitedFilesFromUrl(self, url):
        """getCitedFilesFromUrl returns a list of citations to individual Madrigal file from a group id url
        as found in a publication
        """
        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.strip().split("\n")

        mainUrl.close()

        return page

    def createCitationGroupFromList(
        self, citationList, user_fullname, user_email, user_affiliation, useLocal=False
    ):
        """createCitationGroupFromList creates a new group citation given an input
        list of citations in the form:
        'https://w3id.org/cedar?experiment_list=experiments/2014/mlh/16mar13&file_list=mlh130316g.007.hdf5'
        or 'experiments/2014/mlh/16mar13&file_list=mlh130316g.007.hdf5'

        Inputs:

            citationList - list of citations

            user_fullname - full name of user making request

            user_email - email address of user making request

            user_affiliation - affiliation of user making request

            useLocal - if False (the default) use main CEDAR cite for official creation of citation.  If
                True, use site set in init.

        Returns:

            group citation (string)
        """
        if not useLocal:
            url = "https://cedar.openmadrigal.org/createCitationGroupWithList?"
        else:
            url = urllib.parse.urljoin(self.cgiurl, "createCitationGroupWithList") + "?"
        user_fullname = user_fullname.replace(" ", "+").strip()
        url += "user_fullname=%s&" % (user_fullname)
        user_email = user_email.strip()
        url += "user_email=%s&" % (user_email)
        user_affiliation = user_affiliation.replace(" ", "+").strip()
        url += "user_affiliation=%s&" % (user_affiliation)
        citationDict = {"url": citationList}

        # read main url
        try:
            data = urllib.parse.urlencode(citationDict).encode()
            req = urllib.request.Request(url, data=data)
            mainUrl = urllib.request.urlopen(req, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.strip()

        mainUrl.close()

        return page

    def getCitationListFromFilters(
        self,
        startDate,
        endDate,
        inst=None,
        kindat=None,
        seasonalStartDate=None,
        seasonalEndDate=None,
        includeNonDefault=False,
        expName=None,
        excludeExpName=None,
        fileDesc=None,
    ):
        """getCitationListFromFilters returns a list of citations using filters similar to globalDownload.
        Result can then be used to create citation group using createCitationGroupFromList

        Inputs:
                startDate: start datetime to filter experiments before in YYYY-MM-DD (required)
                endDate: end datetime to filter experiments after  in YYYY-MM-DD (required)
                inst: a list of instrument codes or names. If None, all instruments used.
                    For names fnmatch will be used
                kindat: a list of kind of data codes or names. If None, all kindats used.
                    For names fnmatch will be used
                seasonalStartDate: in form MM/DD, rejects all days earlier in year. If None,
                    implies 01/01
                seasonalEndDate: in form MM/DD, rejects all days later in year. If None,
                    implies 12/31
                includeNonDefault: if set, include realtime files when there are no default. If False,
                    implies only default files.
                expName: string - filter experiments by the experiment name.  fnmatch rules
                    If None, no filtering by experiment name.
                excludeExpName: string - exclude experiments by the experiment name.  fnmatch rules
                    If None, no excluding experiments by experiment name.
                fileDesc: filter files using input file Description string via fnmatch.
                    If None, no filtering by file name

        Returns a list with all citations in group, which can be used in createCitationGroupFromList
        """
        url = "https://cedar.openmadrigal.org/getCitationGroupWithFilters?"
        # temp only - until cedar updated
        # url = 'http://127.0.0.1:8000/getCitationGroupWithFilters?'
        url += "startDate=%s&" % (startDate.strftime("%Y-%m-%d"))
        url += "endDate=%s&" % (endDate.strftime("%Y-%m-%d"))
        if not inst is None:
            for thisInst in inst:
                url += "inst=%s&" % (urllib.parse.quote_plus(str(thisInst)))
        if not kindat is None:
            for thisKindat in kindat:
                url += "kindat=%s&" % (urllib.parse.quote_plus(str(thisKindat)))
        if not seasonalStartDate is None:
            url += "seasonalStartDate=%s&" % (seasonalStartDate.strip())
        if not seasonalEndDate is None:
            url += "seasonalEndDate=%s&" % (seasonalEndDate.strip())
        if includeNonDefault:
            url += "includeNonDefault=True&"
        if not expName is None:
            url += "expName=%s&" % (urllib.parse.quote_plus(expName.strip()))
        if not excludeExpName is None:
            url += "excludeExpName=%s&" % (
                urllib.parse.quote_plus(excludeExpName.strip())
            )
        if not fileDesc is None:
            url += "fileDesc=%s&" % (urllib.parse.quote_plus(fileDesc.strip()))

        # read main url
        try:
            mainUrl = urllib.request.urlopen(url, timeout=TIMEOUT2)
        except:
            raise ValueError("unable to open url " + str(url))

        page = mainUrl.read().decode("utf8")
        page = page.strip().split("\n")

        mainUrl.close()

        return page

@dataclass
class MadrigalInstrument:
    """MadrigalInstrument is a class that encapsulates information about a Madrigal Instrument.

    Attributes::

        name (string) Example: 'Millstone Hill Incoherent Scatter Radar'
        code (int) Example: 30
        mnemonic (3 char string) Example: 'mlh'
        latitude (double) Example: 45.0
        longitude (double) Example: 110.0
        altitude (double)  Example: 0.015 (km)
        category (string) Example 'Incoherent Scatter Radars'

    Change history:

    Written by "Bill Rideout":mailto:wrideout@haystack.mit.edu  Feb. 10, 2004
    """
    name: str
    code: int
    mnemonic: str
    latitude: float
    longitude: float
    altitude: float
    category: str


@dataclass
class MadrigalExperiment:
    """MadrigalExperiment is a class that encapsulates information about a Madrigal Experiment.

    Attributes::

        id (int) Example: 10000111.  Uniquely identifies experiment.
        realUrl (string) the real url to display this experiment in a web browser.
        url (string) Example: 'http://madrigal.haystack.mit.edu/cgi-bin/madtoc/1997/mlh/03dec97'
            Note: this is a an old url that no longer works, but it is the form stored in the metadata.
            See realUrl attribute for working url.
        name (string) Example: 'Wide Latitude Substorm Study'
        siteid (int) Example: 1
        sitename (string) Example: 'Millstone Hill Observatory'
        instcode (int) Code of instrument. Example: 30
        instname (string) Instrument name. Example: 'Millstone Hill Incoherent Scatter Radar'
        start_dt (datetime) start date and time of experiment in UTC
        end_dt (datetime) end date and time of experiment in UTC

        isLocal - True if a local experiment, False if not
        madrigalUrl - url of Madrigal site.  Used if not a local experiment.
        pi - experiment Principal investigator
        piEmail - experiment Principal investigator's email
        realUrl - working url to experiment (use this instead of url)
        uttimestamp - st_mtime of expDir. None if not supported by the Madrigal site
        access - access code of the experiment (0 if public, 2 if public).  None if not
            supported by the Madrigal site.
        version - version of Madrigal site where data located in form I.I[.I] where I is an integer

    Change history:

    Written by "Bill Rideout":mailto:wrideout@haystack.mit.edu  Feb. 10, 2004

    """
    id: int
    url: str
    name: str
    siteid: int
    sitename: str
    instcode: int
    instname: str
    start_dt: datetime.datetime
    end_dt: datetime.datetime
    isLocal: bool = False
    madrigalUrl: str = ""
    pi: str = ""
    piEmail: str = ""
    realUrl: str = ""
    uttimestamp: datetime.datetime | None = None
    access: int | None = None
    version: str = "2.5"  # default to 2.5,


    def _getRealExperimentUrl(self, version: str) -> str:
        """getRealExperimentUrl is a private method that returns the url used in a web browser to see
        this experiment's page in full data access interface. Uses to create attribute realUrl.

        Inputs:
            version - version of Madrigal site where data located in form I.I[.I] where I is an integer

        real url depends on Madrigal version
        """
        version_list = [int(item) for item in version.split(".")]
        retStr = ""
        index = self.url.find("/madtoc/")
        if version_list[0] == 2:
            retStr += self.url[:index] + "/madExperiment.cgi?exp="
            retStr += self.url[index + 8 :] + "&displayLevel=0&expTitle="
            try:
                retStr += urllib.parse.quote_plus(self.name)  # python3
            except AttributeError:
                retStr += urllib.quote_plus(self.name.encode("utf8"))  # python2
        else:
            retStr += self.url[:index] + "/showExperiment/?experiment_list=%i" % (
                self.id
            )
        return retStr


@dataclass
class MadrigalExperimentFile:
    """MadrigalExperimentFile is a class that encapsulates information about a Madrigal ExperimentFile.

    Attributes::
        name (string) Example '/opt/mdarigal/blah/mlh980120g.001'
        kindat (int) Kindat code.  Example: 3001
        kindatdesc (string) Kindat description: Example 'Basic Derived Parameters'
        category (int) (1=default, 2=variant, 3=history, 4=real-time)
        status (string)('preliminary', 'final', or any other description)
        permission (int)  0 for public, 1 for private
        expId - experiment id of the experiment this MadrigalExperimentFile belongs in
        doi - digital object identifier - citable url to file - or None if not found


    Change history:

    Written by "Bill Rideout":mailto:wrideout@haystack.mit.edu  Feb. 10, 2004
    """
    name: str
    kindat: int
    kindatdesc: str
    category: int
    status: str
    permission: int
    expId: int
    doi: str | None = None


@dataclass
class MadrigalParameter:
    """__init__ initializes a MadrigalParameter.

    Inputs::

        mnemonic (string) Example 'dti'
        description (string) Example: "F10.7 Multiday average observed (Ott)"
        isError (int) 1 if error parameter, 0 if not
        units (string) Example "W/m2/Hz"
        isMeasured (int) 1 if measured, 0 if derivable
        category (string) Example: "Time Related Parameter"
        isSure (int) - 1 if parameter can be found for every record, 0 if can only be found for some
        isAddIncrement - 1 if additional increment, 0 if normal, -1 if unknown (this capability
                            only added with Madrigal 2.5)

    Returns: void

    Affects: Initializes all the class member variables.

    Exceptions: If illegal argument passed in.
    """
    mnemonic: str
    description: str
    isError: int
    units: str
    isMeasured: int
    category: str
    isSure: int
    isAddIncrement: int = -1  # default to -1, which means unknown
