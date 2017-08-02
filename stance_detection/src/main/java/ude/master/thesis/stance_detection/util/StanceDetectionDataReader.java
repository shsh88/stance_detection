package ude.master.thesis.stance_detection.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVReader;

/**
 * This utility class is responsible for reading the data provided by FNC-1 and
 * save it in the corresponding data structures.
 * 
 * The data is only read as it is here / only the strings with no further
 * processing
 * 
 * @author Razan
 *
 */

public class StanceDetectionDataReader {
	//TODO still trainTitleIdMap not filled / not used.. maybe should use title-body map instead
	private Map<String, String> trainTitleIdMap;
	private Map<Integer, String> trainIdBodyMap;
	private List<List<String>> trainStances;

	private HashMap<Integer, String> testIdBodyMap;
	private List<List<String>> testStances;

	private static final String TRAIN_BODIES_CSV_LOCATION = "resources/data/train_bodies.csv";
	private static final String TRAIN_STANCES_CSV_LOCATION = "resources/data/train_stances.csv";

	private static final String TEST_BODIES_CSV_LOCATION = "resources/data/test_data/competition_test_bodies.csv";
	private static final String TEST_STANCES_CSV_LOCATION = "resources/data/test_data/competition_test_stances.csv";

	private boolean defaultLocation;

	/**
	 * By calling the constructor, the data is read and ready to be dealt with
	 * by calling the variables that hold them
	 * 
	 * @throws IOException
	 */
	public StanceDetectionDataReader(boolean setTrainingData, boolean setTestData) throws IOException {
		this.defaultLocation = true;
		if (setTrainingData)
			setTrainingData();
		if (setTestData)
			setTestData();
	}

	/**
	 * Set testDataLocation to "" if you don't want to use test data
	 * 
	 * @param setTrainingData
	 * @param setTestData
	 * @param trainingDataLoc
	 * @param testDataLocation
	 * @throws IOException
	 */
	public StanceDetectionDataReader(boolean setTrainingData, boolean setTestData, String trainingStanceDataLoc,
			String trainingBodiesDataLoc, String testStanceDataLocation, String testBodiesDataLocation)
			throws IOException {
		this.defaultLocation = false;
		if (setTrainingData)
			setTrainingData(trainingStanceDataLoc, trainingBodiesDataLoc);
		if (setTestData)
			if (!testBodiesDataLocation.equals("") & !testStanceDataLocation.equals(""))
				setTestData(testStanceDataLocation, testBodiesDataLocation);
			else
				setTestData();
	}

	private void setTestData(String testStanceDataLocation, String testBodiesDataLocation)
			throws FileNotFoundException, IOException {
		testIdBodyMap = readInIdBodiesMap(new File(testBodiesDataLocation));
		testStances = readStances(new File(testStanceDataLocation));
	}

	private void setTrainingData(String trainingStanceDataLoc, String trainingBodiesDataLoc)
			throws FileNotFoundException, IOException {
		trainIdBodyMap = readInIdBodiesMap(new File(trainingBodiesDataLoc));
		trainStances = readStances(new File(trainingStanceDataLoc));
	}

	/**
	 * Create a mapping for Id & body and Create a list of each line in the
	 * stances file
	 * 
	 * @throws IOException
	 */
	private void setTrainingData() throws IOException {
		trainIdBodyMap = readInIdBodiesMap(new File(TRAIN_BODIES_CSV_LOCATION));
		trainStances = readStances(new File(TRAIN_STANCES_CSV_LOCATION));

	}

	private void setTestData() throws IOException {
		testIdBodyMap = readInIdBodiesMap(new File(TEST_BODIES_CSV_LOCATION));
		testStances = readStances(new File(TEST_STANCES_CSV_LOCATION));
	}

	private HashMap<Integer, String> readInIdBodiesMap(File bodiesFile) {
		HashMap<Integer, String> bodyMap = new HashMap<>(100, 100);
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(bodiesFile));
			String[] line;
			line = reader.readNext();
			while ((line = reader.readNext()) != null) {
				bodyMap.put(Integer.valueOf(line[0]), line[1]);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return bodyMap;
	}

	private List<List<String>> readStances(File stancesFile) throws FileNotFoundException, IOException {
		CSVReader stancesReader = new CSVReader(new FileReader(stancesFile));
		String[] stancesline;
		List<List<String>> stances = new ArrayList<>();
		stancesReader.readNext();
		System.out.println(stancesReader.getLinesRead());
		while ((stancesline = stancesReader.readNext()) != null) {
			List<String> record = new ArrayList<>();
			record.add(stancesline[0]);
			record.add(stancesline[1]);
			record.add(stancesline[2]);
			stances.add(record);
		}

		stancesReader.close();
		return stances;
	}

	public Map<String, String> getTrainTitleIdMap() {
		return trainTitleIdMap;
	}

	public void setTrainTitleIdMap(Map<String, String> titleIdMap) {
		this.trainTitleIdMap = titleIdMap;
	}

	public Map<Integer, String> getTrainIdBodyMap() {
		return trainIdBodyMap;
	}

	public void setTrainIdBodyMap(Map<Integer, String> idBodyMap) {
		this.trainIdBodyMap = idBodyMap;
	}

	public List<List<String>> getTrainStances() {
		return trainStances;
	}

	public void setTrainStances(List<List<String>> stances) {
		this.trainStances = stances;
	}

	public HashMap<Integer, String> getTestIdBodyMap() {
		return testIdBodyMap;
	}

	public void setTestIdBodyMap(HashMap<Integer, String> testIdBodyMap) {
		this.testIdBodyMap = testIdBodyMap;
	}

	public List<List<String>> getTestStances() {
		return testStances;
	}

	public void setTestStances(List<List<String>> testStances) {
		this.testStances = testStances;
	}

	public boolean isDefaultLocation() {
		return defaultLocation;
	}

	public void setDefaultLocation(boolean defaultLocation) {
		this.defaultLocation = defaultLocation;
	}

}
