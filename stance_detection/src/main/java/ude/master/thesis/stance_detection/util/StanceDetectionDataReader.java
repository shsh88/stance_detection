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
 * The data is only read as it is here / only the strings with no further processing
 * 
 * @author Razan
 *
 */

public class StanceDetectionDataReader {
	private Map<String, String> titleIdMap;
	private Map<Integer, String> idBodyMap;
	private List<List<String>> stances;

	private String TRAIN_BODIES_CSV_LOCATION = "resources/data/train_bodies.csv";
	private String TRAIN_STANCES_CSV_LOCATION = "resources/data/train_stances.csv";

	/**
	 * By calling the constructor, the data is read and ready to be dealt with
	 * by calling the variables that hold them
	 * 
	 * @throws IOException
	 */
	public StanceDetectionDataReader() throws IOException {
		setData();
	}

	/**
	 * Create a mapping for Id & body 
	 * and Create a list of each line in the stances file
	 * 
	 * @throws IOException
	 */
	private void setData() throws IOException {
		idBodyMap = readInIdBodiesMap(new File(TRAIN_BODIES_CSV_LOCATION));

		stances = readStances(new File(TRAIN_STANCES_CSV_LOCATION));
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

	public Map<String, String> getTitleIdMap() {
		return titleIdMap;
	}

	public void setTitleIdMap(Map<String, String> titleIdMap) {
		this.titleIdMap = titleIdMap;
	}

	public Map<Integer, String> getIdBodyMap() {
		return idBodyMap;
	}

	public void setIdBodyMap(Map<Integer, String> idBodyMap) {
		this.idBodyMap = idBodyMap;
	}

	public List<List<String>> getStances() {
		return stances;
	}

	public void setStances(List<List<String>> stances) {
		this.stances = stances;
	}

}
