package ude.master.thesis.stance_detection.util;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

public class ProjectPaths {
	public static final String TRAIN_STANCES_PREPROCESSED = "resources/data/train_stances_preprocessed.csv";
	public static final String TRAIN_STANCES = "resources/data/train_stances.csv";
	public static final String SUMMARIZED_TRAIN_BODIES = "resources/data/train_bodies_preprocessed_summ.csv";
	public static final String TEST_STANCESS_PREPROCESSED = "resources/data/test_data/test_stances_preprocessed.csv";
	public static final String TEST_STANCESS = "resources/data/test_data/competition_test_stances.csv";
	public static final String SUMMARIZED_TEST_BODIES = "resources/data/test_data/test_bodies_preprocessed_summ.csv";

	// Saved features
	public static final String ROOT_DIST_FEATURE_TRAIN = "C:/thesis_stuff/features/train_features/train_rootdist";
	public static final String ROOT_DIST_FEATURE_TEST = "C:/thesis_stuff/features/test_features/test_rootdist";

	public static final String SVO_FEATURE_TRAIN = "C:/thesis_stuff/features/train_features/train_svo_nosvo_features";
	public static final String SVO_FEATURE_TEST = "C:/thesis_stuff/features/test_features/test_svo_nosvo_features";

	public static final String PPDB_HUNG_FEATURE_TRAIN = "C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords";
	public static final String PPDB_HUNG_FEATURE_TEST = "C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords";

	public static final String NEG_FEATURE_TRAIN = "C:/thesis_stuff/features/train_features/train_neg_features";
	public static final String NEG_FEATURE_TEST = "C:/thesis_stuff/features/test_features/test_neg_features";
	
	public static final String W2V_SIM_ADD_TRAIN = "C:/thesis_stuff/features/train_features/train_w2v_sim_features";
	public static final String W2V_SIM_ADD_TEST = "C:/thesis_stuff/features/test_features/test_w2v_sim_features";
	public static final String ARFF_DATA_PATH = "C:/thesis_stuff/arff/";
	public static final String ARFF_EXTENSION = ".arff";
	public static final String RESULTS_PATH = "C:/thesis_stuff/results/";
	public static final String SCORING_CSV = "C:/thesis_stuff/scoring_CSVs/";
	public static final String TITLES_LEMMAS = "C:/thesis_stuff/help_files/titles_lemmas";
	public static final String BODIES_LEMMAS = "C:/thesis_stuff/help_files/bodies_lemmas";
	public static final String MODEL_PATH = "C:/thesis_stuff/models/";
	public static final String TRAIN_WORD_OVERLAPS_PATH = "C:/thesis_stuff/features/train_features/train_words_overlaps";
	public static final String TEST_WORD_OVERLAPS_PATH = "C:/thesis_stuff/features/test_features/test_words_overlaps";
	public static final String TEST_COOCC_PATH = "C:/thesis_stuff/features/test_features/test_cooc_stop";
	public static final String TRAIN_COOCC_PATH = "C:/thesis_stuff/features/train_features/train_cooc_stop";
	public static final String TRAIN_CGRAMS_PATH = "C:/thesis_stuff/features/train_features/train_cgrams";
	public static final String TEST_CGRAMS_PATH = "C:/thesis_stuff/features/test_features/test_cgrams";
	public static final String TRAIN_NGRAMS_PATH = "C:/thesis_stuff/features/train_features/train_ngrams";
	public static final String TEST_NGRAMS_PATH = "C:/thesis_stuff/features/test_features/test_ngrams";
	public static final String TRAIN_COS_SIM_PATH = "C:/thesis_stuff/features/train_features/train_cossim";
	public static final String TEST_COS_SIM_PATH = "C:/thesis_stuff/features/test_features/test_cossim";
	public static final String TRAIN_COS_SIM_WS_PATH = "C:/thesis_stuff/features/train_features/train_ws_cos_sim";
	public static final String TEST_COS_SIM_WS_PATH = "C:/thesis_stuff/features/test_features/test_ws_cos_sim";
	public static final String TRAIN_LESK_PATH = "C:/thesis_stuff/features/train_features/train_lesk";
	public static final String TEST_LESK_PATH = "C:/thesis_stuff/features/test_features/test_lesk";
	public static final String TRAIN_COS_SIM_qgram5_PATH = "C:/thesis_stuff/features/train_features/train_5qgram_cos_sim";
	public static final String TEST_COS_SIM_qgram5_PATH = "C:/thesis_stuff/features/test_features/test_5qgram_cos_sim";
	public static final String TITLES_POS_MAP_PATH = "C:/thesis_stuff/help_files/titles_pos_map";
	public static final String BODIES_POS_MAP_PATH = "C:/thesis_stuff/help_files/bodies_pos_map";
	public static final String TRAIN_COS_SIM_STRMET_PATH = "C:/thesis_stuff/features/train_features/train_strmet_cos_sim";
	public static final String TEST_COS_SIM_STRMET_PATH = "C:/thesis_stuff/features/test_features/test_strmet_cos_sim";
	public static final String WORDNET_DICT = "C:/Master UDE/thesis/software/WordNet_3.1/dict";
	public static final String TRAIN_HYP_SIM_PATH = "C:/thesis_stuff/features/train_features/train_hyp_sim";
	public static final String TEST_HYP_SIM_PATH = "C:/thesis_stuff/features/test_features/test_hyp_sim";
	
	public static final String TRAIN_LESS_10_BODY = "C:/thesis_stuff/help_files/train_less_10_body.txt";
	public static final String TRAIN_LESS_5_BODY = "C:/thesis_stuff/help_files/train_less_5_body.txt";
	public static final String TRAIN_LESS_2_BODY = "C:/thesis_stuff/help_files/train_less_2_body.txt";
	public static final String TRAIN_TITLES_LENGTH_DATA = "C:/thesis_stuff/help_files/train_title_length_data.txt";
	
	public static final String TEST_LESS_10_BODY = "C:/thesis_stuff/help_files/test_less_10_body.txt";
	public static final String TEST_LESS_5_BODY = "C:/thesis_stuff/help_files/test_less_5_body.txt";
	public static final String TEST_LESS_2_BODY = "C:/thesis_stuff/help_files/test_less_2_body.txt";
	public static final String TEST_TITLES_LENGTH_DATA = "C:/thesis_stuff/help_files/test_title_length_data.txt";
	public static final String TRAIN_eq9_BODY = "C:/thesis_stuff/help_files/train_eq9_body.txt";
	public static final String TRAIN_eq8_BODY = "C:/thesis_stuff/help_files/train_eq8_body.txt";
	public static final String TRAIN_eq7_BODY = "C:/thesis_stuff/help_files/train_eq7_body.txt";
	public static final String TRAIN_eq6_BODY = "C:/thesis_stuff/help_files/train_eq6_body.txt";
	public static final String TEST_eq9_BODY = "C:/thesis_stuff/help_files/test_eq9_body.txt";
	public static final String TEST_eq8_BODY = "C:/thesis_stuff/help_files/test_eq8_body.txt";
	public static final String TEST_eq7_BODY = "C:/thesis_stuff/help_files/test_eq7_body.txt";
	public static final String TEST_eq6_BODY = "C:/thesis_stuff/help_files/test_eq6_body.txt";
	
	
	
	

}
