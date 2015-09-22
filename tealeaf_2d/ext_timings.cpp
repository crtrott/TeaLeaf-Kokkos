#include <iostream>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <string>
#include <stdlib.h>
#include <omp.h>

using namespace std;
const string failureLog = "timingfailures.log";
const string successPrefix = " This test is considered ";
const string probSizePrefix[] = { "x_cells=", "y_cells=", "z_cells=" };
const string timePrefix = " Wall clock ";
const string configGuard = "*tea";
const char comment = '!';
std::string get_clean_string(char* dirtyString);

/*
 *      Very nasty tool for parsing the log file for timing results.
 */
extern "C"
void ext_log_timings_(char* outLog, char* teaOutPath, int* numtasks)
{
    string teaOutPathClean = get_clean_string(teaOutPath);
    string outLogClean = get_clean_string(outLog);

	ifstream teaOutFile(teaOutPathClean.c_str());

	if(!teaOutFile.is_open())
	{
		cerr << "Could not open " << teaOutPathClean << " for parsing results." << endl;
		return;
	}

	// Get the current working directory
	char buf[1024];
	if(getcwd(buf, sizeof(buf))==NULL)
	{
		cerr << "Could not get working directory." << endl;
		return;
	}

	string cwd(buf);
	int sep = cwd.rfind("/");
	cwd = cwd.substr(sep+1, cwd.length()-sep);

	char* path;
	path = getenv("OMP_NUM_THREADS");
	int numthreads = (path == NULL) ?
		0 : atoi(path);

	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );

	ostringstream oss;
	oss << now->tm_mday << '-'
		<< (now->tm_mon + 1) << '-'
		<< (now->tm_year + 1900) << " "
		<< now->tm_hour << ":"
		<< now->tm_min << ":"
		<< now->tm_sec;

	oss << " " << teaOutPathClean << " " << cwd << " " 
		<< *numtasks << " " << numthreads;

	// Get the problem size
	int probSize[2] = {0,0};
	string line,temp;
	while(getline(teaOutFile, line))
	{
		for(int ii = 0; ii != 2; ++ii)
		{
			size_t pl = probSizePrefix[ii].length();
			if(line.length() > pl && line.substr(0,pl) == probSizePrefix[ii])
			{
				string probSizeWord = line.substr(pl);
				probSize[ii] = atoi(probSizeWord.c_str());
			}
		}

		if(probSize[0] > 0 && probSize[1] > 0) break;
	}

	oss << " " << probSize[0] << "x" << probSize[1];

	// Get the solver type
	while(getline(teaOutFile, line))
	{
		istringstream iss(line);
		iss >> temp;

		if(temp[0] == comment) continue;

		if(temp == "tl_use_jacobi" || temp == "tl_use_chebyshev" 
				|| temp == "tl_use_cg" || temp == "tl_use_ppcg")
		{
			oss << " " << temp.substr(7, temp.length()-7);
			break;
		}
	}

	// Get the success
	while(getline(teaOutFile, line))
	{
		size_t pl = successPrefix.length();

		if(line.length() > pl && line.substr(0,pl) == successPrefix)
		{
			oss << " " << line.substr(pl);
			break;
		}
	}

	// Get the time
	while(getline(teaOutFile, line))
	{
		size_t pl = timePrefix.length();
		if(line.length() > pl && line.substr(0,pl) == timePrefix)
		{
			string timeWord = line.substr(pl);
			double time = atof(timeWord.c_str());
			oss << " " << setprecision(12) << time;
			break;
		}
	}

	ofstream log(outLogClean.c_str(), ios::app);
	string logLine = oss.str();

	if(!log.is_open())
	{
		cerr << "Could not open timing log, storing locally!" << endl;
		ofstream localLog(failureLog.c_str(), ios::app);
		localLog << logLine << endl;
	}
	else
	{
		cout << "Successfully wrote to global timing log." << endl;
		log << logLine << endl;
	}

}

std::string get_clean_string(char* dirtyString)
{
	string temp(dirtyString);
	size_t endpos = temp.find_first_of(' ');
	if(endpos != string::npos)
	{
		return temp.substr(0, endpos);
	}
	return temp;
}
