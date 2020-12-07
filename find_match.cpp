#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std; 

// Spliting string by delimiter
vector<string> split(const string& str, char delimiter) {
    vector<string> output;
    stringstream ss(str);
    string temp;
 
    while (getline(ss, temp, delimiter)) {
        output.push_back(temp);
    }
 
    return output;
}

// Print string vector (for debuging)
void print_vector(vector<string> v) {
    vector<string> path = v;
    for (vector<string>::iterator i = path.begin(); i != path.end(); ++i)
        cout << *i << "*";
    cout << "||" <<endl;
}

// Counting matches
int count_matches(vector<pair<int,int>> input, vector<pair<int,int>> target) {
    int flag = 0;
    int i = 0;
    int matches = 0;

    for (i=0 ; i < input.size() ; i++) {
        if (find(target.begin(), target.end(), input[i]) != target.end()) {
            matches ++;
        }
    }

    return matches;
}

// Make vector<pair<int,int>> wiht vectors
vector<pair<int,int>> str_to_int(vector<string> &str_vec, int start, bool is_oracle) {
    vector<pair<int,int>> coord;
    string temp = "";

    for (vector<string>::iterator i = str_vec.begin()+start ; i != str_vec.end() ; ++i) {
        temp += *i;
        vector<string> str = split(temp, ' ');
        int x = atoi(str[0].c_str());
        int y = atoi(str[1].c_str());
        coord.push_back(make_pair(x,y));

        // Oracle need to have more generous judgement for coordinate matching
        if (is_oracle) {
            coord.push_back(make_pair(x+1,y));
            coord.push_back(make_pair(x,y+1));
            coord.push_back(make_pair(x-1,y));
            coord.push_back(make_pair(x,y-1));
        }

        temp = "";
    }
    return coord;
}

// Find matches
string find_matches(const string &input, const string &oracle) {
    vector<string> input_vec = split(input, ',');
    vector<string> oracle_vec = split(oracle, ',');

    // if the line is not a data, return.
    if (input_vec.size() < 2) {
        return input;
    }

    string result = input_vec[0];   // Data Name
    result += ",";                 
    result += input_vec[1];         // Data's corner count (predicted)
    result += ",";
    result += oracle_vec[1];        // Data's corner count (GT)

    vector<pair<int,int>> input_points = str_to_int(input_vec, 2, false);
    vector<pair<int,int>> oracle_points = str_to_int(oracle_vec, 2, true);

    int matches = count_matches(input_points, oracle_points);

    result += ",";
    result += to_string(matches);   // Matching corner count

    // print_vector(input_vec);
    // print_vector(oracle_vec);

    return result;
}

/*
    The goal of this cpp code is to find matching corner counts.
    This code assumes that the order of data in prediction and GT are same.
*/
int main (int argc, char** argv) {
	string in_line;
    string oracle_line;
    ifstream fileInput;
    ifstream fileOracle;

    fileInput.open(argv[1]);
    fileOracle.open(argv[2]);

    if (fileInput.is_open() && fileOracle.is_open()) {
        while(!fileInput.eof() && !fileOracle.eof()) {
            getline(fileInput, in_line);
            getline(fileOracle, oracle_line);

            cout << find_matches(in_line, oracle_line) << endl;
        }
        fileInput.close();
    }

    return 0;
}

