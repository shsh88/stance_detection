package ude.master.thesis.stance_detection.util;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

public class ProjectPaths {
	public static final String TRAIN_STANCES_PREPROCESSED = "resources/data/train_stances_preprocessed.csv";
	public static final String TRAIN_STANCES_LESS2_PREPROCESSED = "resources/data/train_stances_less2_preprocessed.csv";
	
	public static final String TRAIN_STANCES = "resources/data/train_stances.csv";
	public static final String SUMMARIZED_TRAIN_BODIES = "resources/data/train_bodies_preprocessed_summ.csv";
	public static final String TEST_STANCESS_PREPROCESSED = "resources/data/test_data/test_stances_preprocessed.csv";
	public static final String TEST_STANCESS_LESS2_PREPROCESSED = "resources/data/test_data/test_stances_less2_preprocessed.csv";
	
	public static final String TEST_STANCESS = "resources/data/test_data/competition_test_stances.csv";
	public static final String SUMMARIZED_TEST_BODIES = "resources/data/test_data/test_bodies_preprocessed_summ.csv";

	public static final String SUMMARIZED_TRAIN_BODIES2 = "resources/data/train_bodies_preprocessed_summ2.csv";
	public static final String SUMMARIZED_TEST_BODIES2 = "resources/data/test_data/test_bodies_preprocessed_summ2.csv";
	
	public static final String MAP_SUMMARIZED_TRAIN_BODIES2 = "C:/thesis_stuff/data_maps/train_bodies_preprocessed_summ2";
	public static final String MAP_SUMMARIZED_TEST_BODIES2 = "C:/thesis_stuff/data_maps/test_bodies_preprocessed_summ2";
	// Saved features
	public static final String ROOT_DIST_FEATURE_TRAIN = "C:/thesis_stuff/features/train_features/train_rootdist";
	public static final String ROOT_DIST_FEATURE_TEST = "C:/thesis_stuff/features/test_features/test_rootdist";
	
	public static final String CSV_ROOT_DIST_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_rootdist2.csv";
	public static final String CSV_ROOT_DIST_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_rootdist2.csv";

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
	public static final String ARGS_BODIES_LEMMAS = "C:/thesis_stuff/help_files/args_bodies_lemmas";
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
	
	public static final String TRAIN_LESS_10_ARGBODY = "C:/thesis_stuff/help_files/train_less_10_argbody.txt";
	public static final String TRAIN_LESS_5_ARGBODY = "C:/thesis_stuff/help_files/train_less_5_argbody.txt";
	public static final String TRAIN_LESS_2_ARGBODY = "C:/thesis_stuff/help_files/train_less_2_argbody.txt";
	public static final String TRAIN_0ARG_ARGBODY = "C:/thesis_stuff/help_files/train_0args_argbody.txt";
	
	public static final String TEST_LESS_10_ARGBODY = "C:/thesis_stuff/help_files/test_less_10_argbody.txt";
	public static final String TEST_LESS_5_ARGBODY = "C:/thesis_stuff/help_files/test_less_5_argbody.txt";
	public static final String TEST_LESS_2_ARGBODY = "C:/thesis_stuff/help_files/test_less_2_argbody.txt";
	public static final String TEST_0ARG_ARGBODY = "C:/thesis_stuff/help_files/test_0args_argbody.txt";
	public static final String TRAIN_eq9_ARGBODY = "C:/thesis_stuff/help_files/train_eq9_argbody.txt";
	public static final String TRAIN_eq8_ARGBODY = "C:/thesis_stuff/help_files/train_eq8_argbody.txt";
	public static final String TRAIN_eq7_ARGBODY = "C:/thesis_stuff/help_files/train_eq7_argbody.txt";
	public static final String TRAIN_eq6_ARGBODY = "C:/thesis_stuff/help_files/train_eq6_argbody.txt";
	public static final String TEST_eq9_ARGBODY = "C:/thesis_stuff/help_files/test_eq9_argbody.txt";
	public static final String TEST_eq8_ARGBODY = "C:/thesis_stuff/help_files/test_eq8_argbody.txt";
	public static final String TEST_eq7_ARGBODY = "C:/thesis_stuff/help_files/test_eq7_argbody.txt";
	public static final String TEST_eq6_ARGBODY = "C:/thesis_stuff/help_files/test_eq6_argbody.txt";
	
	public static final String TRAIN_BODIES_DEPS2 = "C:/thesis_stuff/help_files/train_bodies_deps2";
	public static final String TRAIN_TITLES_DEPS2 = "C:/thesis_stuff/help_files/train_titles_deps2";
	public static final String TEST_BODIES_DEPS2 = "C:/thesis_stuff/help_files/test_bodies_deps2";
	public static final String TEST_TITLES_DEPS2 = "C:/thesis_stuff/help_files/test_titles_deps2";
	public static final String CSV_PPDB_HUNG_SCORES_IDXS_TRAIN2 = "C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords2.csv";
	public static final String CSV_PPDB_HUNG_SCORES_IDXS_TEST2 = "C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords2.csv";
	public static final String PPDB_HUNG_SCORES_IDXS_TRAIN2 = "C:/thesis_stuff/features/train_features/map_train_hung_ppdb_with_stopwords2";
	public static final String PPDB_HUNG_SCORES_IDXS_TEST2 = "C:/thesis_stuff/features/test_features/map_test_hung_ppdb_with_stopwords2";
	public static final String CSV_NEG_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_neg_features2.csv";
	public static final String CSV_NEG_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_neg_features2.csv";
	public static final String TITLES_SVO_TRAIN_TEST = "C:/thesis_stuff/help_files/titles_svos_train_test";
	public static final String BODIES_SVO_TRAIN_TEST2 = "C:/thesis_stuff/help_files/bodies_svos_train_test2";
	public static final String CSV_SVO_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_svo_features2.csv";
	public static final String CSV_SVO_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_svo_features2.csv";
	public static final String CSV_SUMMED_SVO_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_summed_svo_features2.csv";
	public static final String CSV_SUMMED_SVO_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_summed_svo_features2.csv";
	public static final String BODIES_W2V_SUM2 = "C:/thesis_stuff/help_files/bodies_sum_word2vec2";
	public static final String TITLES_W2V_SUM2 = "C:/thesis_stuff/help_files/titles_sum_word2vec2";
	public static final String CSV_TRAIN_W2V_SUM_SIM_PARTS2 = "C:/thesis_stuff/features/train_features/train_w2v_sim_features2.csv";
	public static final String CSV_TEST_W2V_SUM_SIM_PARTS2 = "C:/thesis_stuff/features/test_features/test_w2v_sim_features2.csv";
	public static final String ROOT_DIST_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_rootdist2";
	public static final String ROOT_DIST_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_rootdist2";
	public static final String PPDB_HUNG_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords2";
	public static final String PPDB_HUNG_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords2";
	public static final String NEG_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_neg_features2";
	public static final String NEG_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_neg_features2";
	public static final String SVO_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_svo_features2";
	public static final String SVO_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_svo_features2";
	public static final String SUMMED_SVO_FEATURE_TRAIN2 = "C:/thesis_stuff/features/train_features/train_summed_svo_features2";
	public static final String SUMMED_SVO_FEATURE_TEST2 = "C:/thesis_stuff/features/test_features/test_summed_svo_features2";
	public static final String W2V_SIM_ADD_TRAIN2 = "C:/thesis_stuff/features/train_features/train_w2v_sim_features2";
	public static final String W2V_SIM_ADD_TEST2 = "C:/thesis_stuff/features/test_features/test_w2v_sim_features2";
	
	public static final String TRAIN_WORD_OVERLAPS_PATH2 = "C:/thesis_stuff/features/train_features/train_words_overlaps2";
	public static final String TEST_WORD_OVERLAPS_PATH2 = "C:/thesis_stuff/features/test_features/test_words_overlaps2";
	public static final String TRAIN_COOCC_PATH2 = "C:/thesis_stuff/features/train_features/train_cooc_stop2";
	public static final String TEST_COOCC_PATH2 = "C:/thesis_stuff/features/test_features/test_cooc_stop2";
	public static final String TRAIN_CGRAMS_PATH2 = "C:/thesis_stuff/features/train_features/train_cgrams2";
	public static final String TEST_CGRAMS_PATH2 = "C:/thesis_stuff/features/test_features/test_cgrams2";
	public static final String TRAIN_NGRAMS_PATH2 = "C:/thesis_stuff/features/train_features/train_ngrams2";
	public static final String TEST_NGRAMS_PATH2 = "C:/thesis_stuff/features/test_features/test_ngrams2";
	
	public static final String TRAIN_COS_SIM_STRMET_PATH2 = "C:/thesis_stuff/features/train_features/train_strmet_cos_sim2";
	public static final String TEST_COS_SIM_STRMET_PATH2 = "C:/thesis_stuff/features/test_features/test_strmet_cos_sim2";
	public static final String TRAIN_LESK_PATH2 = "C:/thesis_stuff/features/train_features/train_lesk2";
	public static final String TEST_LESK_PATH2 = "C:/thesis_stuff/features/test_features/test_lesk2";
	
	
	public static final String BODIES_POS_MAP_PATH2 = "C:/thesis_stuff/help_files/bodies_pos_map2";
	public static final String TRAIN_HYP_SIM_PATH2 = "C:/thesis_stuff/features/train_features/train_hyp_sim2";
	public static final String TEST_HYP_SIM_PATH2 = "C:/thesis_stuff/features/test_features/test_hyp_sim2";
	public static final String PREPROCESSED_BODIES_TEST = "resources/data/test_data/test_bodies_preprocessed.csv";
	public static final String PREPROCESSED_BODIES_TRAIN = "resources/data/train_bodies_preprocessed.csv";
	public static final String ARGUMENTED_BODIES_FILES = "C:/thesis_stuff/tool/ArgMiningTool/src/data/updated";
	
	public static final String ARGUMENTED_BODIES_TRAIN = "resources/data/train_bodies_argumented.csv";
	public static final String ARGUMENTED_BODIES_TEST = "resources/data/test_data/test_bodies_argumented.csv";
	
	public static final String TRAIN_ARG_BODIES_DEPS = "C:/thesis_stuff/help_files/train_arg_bodies_deps";
	public static final String TEST_ARG_BODIES_DEPS = "C:/thesis_stuff/help_files/test_arg_bodies_deps";

	public static final String CSV_ROOT_DIST_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_rootdist_arg.csv";
	public static final String CSV_ROOT_DIST_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_rootdist_arg.csv";

	public static final String ROOT_DIST_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_rootdist_arg";
	public static final String ROOT_DIST_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_rootdist_arg";

	public static final String CSV_PPDB_HUNG_SCORES_IDXS_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_hung_ppdb_with_stopwords.csv";
	public static final String CSV_PPDB_HUNG_SCORES_IDXS_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_hung_ppdb_with_stopwords.csv";
	public static final String PPDB_HUNG_SCORES_IDXS_ARG_TRAIN = "C:/thesis_stuff/features/train_features/map_train_arg_hung_ppdb_with_stopwords";
	public static final String PPDB_HUNG_SCORES_IDXS_ARG_TEST = "C:/thesis_stuff/features/test_features/map_test_arg_hung_ppdb_with_stopwords";
	public static final String PPDB_HUNG_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_hung_ppdb_with_stopwords";
	public static final String PPDB_HUNG_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_hung_ppdb_with_stopwords";
	
	public static final String BODIES_SVO_ARG_TRAIN_TEST = "C:/thesis_stuff/help_files/bodies_svos_train_arg_test";
	public static final String CSV_SUMMED_SVO_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_summed_svo_features.csv";
	public static final String CSV_SUMMED_SVO_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_summed_svo_features.csv";
	public static final String SUMMED_SVO_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_summed_svo_features";
	public static final String SUMMED_SVO_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_summed_svo_features";

	public static final String CSV_NEG_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_neg_features.csv";
	public static final String CSV_NEG_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_neg_features.csv";

	public static final String BODIES_ARG_W2V_SUM = "C:/thesis_stuff/help_files/bodies_arg_sum_word2vec";
	public static final String TITLES_ARG_W2V_SUM = "C:/thesis_stuff/help_files/titles_arg_sum_word2vec";
	public static final String CSV_TRAIN_ARG_W2V_SUM_SIM = "C:/thesis_stuff/features/train_features/train_arg_w2v_sim_features.csv";
	public static final String CSV_TEST_ARG_W2V_SUM_SIM = "C:/thesis_stuff/features/test_features/test_arg_w2v_sim_features.csv";
	
	public static final String TRAIN_ARG_WORD_OVERLAPS_PATH = "C:/thesis_stuff/features/train_features/train_arg_words_overlaps";
	public static final String TEST_ARG_WORD_OVERLAPS_PATH = "C:/thesis_stuff/features/test_features/test_arg_words_overlaps";
	public static final String TEST_ARG_COOCC_PATH = "C:/thesis_stuff/features/test_features/test_arg_cooc_stop";
	public static final String TRAIN_ARG_COOCC_PATH = "C:/thesis_stuff/features/train_features/train_arg_cooc_stop";
	public static final String TRAIN_ARG_CGRAMS_PATH = "C:/thesis_stuff/features/train_features/train_arg_cgrams";
	public static final String TEST_ARG_CGRAMS_PATH = "C:/thesis_stuff/features/test_features/test_arg_cgrams";
	public static final String TRAIN_ARG_NGRAMS_PATH = "C:/thesis_stuff/features/train_features/train_arg_ngrams";
	public static final String TEST_ARG_NGRAMS_PATH = "C:/thesis_stuff/features/test_features/test_arg_ngrams";
	public static final String TRAIN_ARG_COS_SIM_PATH = "C:/thesis_stuff/features/train_features/train_arg_cossim";
	public static final String TEST_ARG_COS_SIM_PATH = "C:/thesis_stuff/features/test_features/test_arg_cossim";
	public static final String TRAIN_ARG_COS_SIM_WS_PATH = "C:/thesis_stuff/features/train_features/train_arg_ws_cos_sim";
	public static final String TEST_ARG_COS_SIM_WS_PATH = "C:/thesis_stuff/features/test_features/test_arg_ws_cos_sim";

	public static final String CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_hung_sppdb_with_stopwords.csv";
	public static final String CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_hung_sppdb_with_stopwords.csv";
	public static final String SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN = "C:/thesis_stuff/features/train_features/map_train_arg_hung_sppdb_with_stopwords";
	public static final String SPPDB_HUNG_SCORES_IDXS_ARG_TEST = "C:/thesis_stuff/features/test_features/map_test_arg_hung_sppdb_with_stopwords";
	public static final String SPPDB_HUNG_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_hung_sppdb_with_stopwords";
	public static final String SPPDB_HUNG_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_hung_sppdb_with_stopwords";

	public static final String TRAIN_ARG_COS_SIM_STRMET_PATH = "C:/thesis_stuff/features/train_features/train_arg_strmet_cos_sim";
	public static final String TEST_ARG_COS_SIM_STRMET_PATH = "C:/thesis_stuff/features/test_features/test_arg_strmet_cos_sim";
	public static final String TRAIN_ARG_LESK_PATH = "C:/thesis_stuff/features/train_features/train_arg_lesk";
	public static final String TEST_ARG_LESK_PATH = "C:/thesis_stuff/features/test_features/test_arg_lesk";
	
	public static final String BODIES_ARGS_POS_MAP_PATH = "C:/thesis_stuff/help_files/bodies_args_pos_map";

	public static final String W2V_SIM_ADD_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_w2v_sim_features";
	public static final String W2V_SIM_ADD_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_w2v_sim_features";

	public static final String NEG_FEATURE_ARG_TRAIN = "C:/thesis_stuff/features/train_features/train_arg_neg_features";
	public static final String NEG_FEATURE_ARG_TEST = "C:/thesis_stuff/features/test_features/test_arg_neg_features";

	public static final String TRAIN_ARG_HYP_SIM_PATH = "C:/thesis_stuff/features/train_features/train_arg_hyp_sim";
	public static final String TEST_ARG_HYP_SIM_PATH = "C:/thesis_stuff/features/test_features/test_arg_hyp_sim";

	
	
	public static final String SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS = "resources/data/train_bodies_preprocessed_parts_noargs.csv";
	public static final String SUMMARIZED_TEST_BODIES_PARTS_NOARGS = "resources/data/test_data/test_bodies_preprocessed_parts_noargs.csv";
	
	public static final String MAP_SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS = "C:/thesis_stuff/data_maps/train_bodies_preprocessed_parts_noargs";
	public static final String MAP_SUMMARIZED_TEST_BODIES_PARTS_NOARGS = "C:/thesis_stuff/data_maps/test_bodies_preprocessed_parts_noargs";

	public static final String ARGUMENTED_MID_BODIES_TRAIN = "resources/data/train_mid_bodies_argumented.csv";
	public static final String ARGUMENTED_MID_BODIES_TEST = "resources/data/test_data/test_mid_bodies_argumented.csv";

}