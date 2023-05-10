import os
import chromadb
from chromadb.config import Settings
import requests
import json
from dotenv import load_dotenv
load_dotenv()
# set your api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}',
}

data = {
    'input': 'Abap best practice?',
    'model': 'text-embedding-ada-002',
}

response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, data=json.dumps(data))

# The json method returns the json-encoded content of a response, if any.
json_response = response.json()

# Print the json response
print(json_response)

# Save the response to a file
with open('response.json', 'w') as f:
    json.dump(json_response, f)


client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/"
))

collection = client.get_collection(name="test")

# Perform ChromaDB query
results = collection.query(
    query_embeddings=[-0.009597635, -0.008454026, 0.014613594, -0.03231059, -0.013260081, 0.031673644, -0.001927127, 0.007903935, -0.012673801, -0.030312894, 0.013983885, -0.0059062373, 0.023393331, 0.009633825, -0.010473438, 0.016922528, 0.022351054, -0.0066300407, 0.024811987, 0.015590729, -0.019108415, 0.017052813, 0.019658504, -0.025362076, -0.013882552, 0.0134048425, 0.010444486, -0.02620169, 0.0036678745, -0.01647377, 0.042356987, -0.0005206862, -0.0053525274, 0.008330979, -0.0041763466, -0.0003598661, -0.0152288275, 0.0022673146, 0.030486606, 0.011588095, 0.026216166, -0.014280644, 0.009373256, -0.01742919, 0.02007831, 0.010220107, -0.004654057, -0.00033294965, -0.035205804, 0.0051751956, 0.01628558, 0.015836822, -0.031673644, 0.012507326, 0.0037673975, 0.014801783, -0.008852118, 0.010473438, -0.009959538, 0.009040306, -0.005262052, 0.0067168972, -0.020353356, 0.020802114, -0.011646, -0.0024735988, 0.01274618, -0.003202831, 0.00014871902, -0.012557992, 0.01569206, 0.016908051, -0.0179069, 0.007230798, 0.010292487, -0.018818893, -0.005703572, -0.00014736188, -0.006101664, 0.00044468683, 0.0023378856, -0.0024410274, -0.010227344, 0.0046938662, 0.014642546, 0.004031586, 0.005471955, -0.0041401563, 0.00085770723, 0.00071249413, 0.0063658524, 0.021164017, 0.024276372, 0.022597147, -0.009503541, 0.016604055, 0.010162202, 0.017385762, 0.0003361163, -0.0129488455, -0.01455569, -0.0031286408, -0.030197086, -0.007513081, 0.016300056, 0.007983553, 0.005515383, 0.008005268, 0.02122192, -0.022597147, -0.013904267, 0.08216618, 0.0125797065, -0.0047445325, 0.011291335, -0.022076009, 0.029386425, -0.014143121, -0.0092864, 0.0020755068, -0.0044984394, 0.020715259, 0.03659551, -0.0080197435, 0.027996723, -0.0074262246, -0.0030743557, -0.024942271, -0.028865287, -0.017602902, 0.049015976, 0.03453991, -0.002811977, 0.0151998745, 0.00056049536, 0.0030490225, -0.0072235595, 0.012586944, -0.011530191, -0.02310381, 0.024131611, 0.023552569, -0.020758687, 0.0045563434, -0.008309265, 0.023480188, 0.027663773, 0.001278418, 0.0013137035, -0.014562928, -0.008649453, -0.009170591, -0.005251195, -0.013158749, -0.026317498, -0.010154964, -0.01293437, 0.0076144137, -0.049768735, 0.008634977, -0.0151998745, 0.04739466, 0.0055660494, 0.016386913, 0.006344138, 0.020860018, 0.013730554, -0.011262383, -0.0034760667, -0.00079889817, -0.001628558, -0.007657842, 0.0041727275, 0.0047626277, -0.0055479542, 0.0010983719, 0.009496303, -0.0054936693, 0.012442184, 0.003850635, -0.002741406, 0.0029965467, 0.016126344, 0.024754083, -0.01964403, 0.01783452, 0.0030381654, 0.0179069, -0.00012587396, -0.0018511276, 0.007715746, 0.018601751, -0.0047988174, -0.01602501, -0.6513074, -0.015894726, 0.005667382, -0.04160423, 0.039085392, 0.0065829935, 0.01919527, -0.0016538912, -0.010748483, 0.02174306, -0.004245108, 0.017472617, 0.016444817, -0.019137366, 0.0024663606, -0.032889634, 0.017733188, -0.019803265, 0.0017307953, 0.010256297, -0.0013978457, 0.017530523, -0.010212868, 0.013643697, 0.0006378519, -0.01809509, 0.015648633, -0.028315196, 0.0015778918, 0.010263534, -0.0022763622, 0.02830072, -8.516001e-05, -0.0010377534, 0.05958351, 0.00663366, -0.020686306, 0.010451724, -0.01016944, 0.03393191, -0.009489065, -0.015301207, 0.010791911, 0.014895878, -0.0013281795, -0.009474589, 0.022264197, -0.012594182, -0.00854812, 0.009720682, -0.008815927, 0.0046504377, -0.035495326, 0.004520153, 0.010147726, 0.0024084563, 0.007114989, -0.003923015, 0.017559474, 0.024435608, 0.013527889, 0.010661626, -0.0020338881, -0.026766255, -0.017675282, -0.0073466063, -0.011139337, -0.0075782235, 0.0060437596, -0.044817917, -0.0002963071, 0.013317985, -0.003438067, -0.003416353, 0.020367833, 0.008338217, -0.010372105, -0.005461098, -0.025839787, -0.008287551, -0.0067856587, -0.0013933219, -0.03231059, -0.02214839, 0.008786975, 0.04985559, -0.033439726, -0.014049027, 0.0046504377, 0.0010241821, 0.027880913, 0.0043174885, 0.0010857054, -0.025564741, 0.040388238, 0.024479037, -0.010227344, -0.012753419, -0.0023957898, -0.011479525, -0.027287394, 0.016994908, -0.0025296935, 0.01499721, 0.029791756, -0.027490059, -0.009199543, 0.0019488411, 0.04163318, -0.0021569347, -0.00034471144, -0.025043603, -0.009228496, -0.014425405, 0.019774314, -0.03338182, 0.0051100533, 0.0054466217, 0.00674223, -0.01702386, -0.008490216, -0.0072018453, 0.011957235, -0.004234251, 0.005099196, 0.012825799, 0.022104962, -0.01094391, -0.010951148, -0.027576916, 0.011501239, 0.011769046, 0.0021189349, -0.0058302376, -0.0043247263, -0.02560817, -0.01473664, 0.0010766578, -0.005747, 0.0026780732, -0.002019412, -0.0063875667, 0.01293437, 0.003264354, -0.013614745, -0.03054451, -0.006756706, 0.018587276, -0.009293638, 0.015721014, 0.0010875149, -0.0147149265, -0.00611614, 0.027591392, -0.0016891765, -0.02107716, 0.0033910198, -0.051361103, -0.025448933, -0.011696666, 0.016777767, 0.019253176, -0.017009383, -0.00766508, -0.0044948203, -0.008461264, 0.008077648, 0.0056058588, -0.03054451, -0.031528883, 0.004288536, -0.02877843, -0.008888308, 0.030862983, -0.028402053, -0.00093280186, -0.0133469375, -0.001927127, 0.02446456, -0.0043102503, 0.01809509, 0.01381741, -0.0061197593, 0.002513408, 0.03164469, 0.0038253018, -0.01297056, 0.016531674, -0.00589538, 0.01997698, -0.01573549, 0.017631855, 0.00061659014, 0.017400239, -0.018485943, 0.01931108, -0.0009771348, 0.029053476, -0.0061559496, 0.02148249, 0.03199212, 0.013643697, 0.038709015, -0.00563843, 0.0027956914, -0.024956748, 0.018268801, -0.027359774, 0.03659551, -0.002115316, -0.0151998745, 0.0061668064, 0.013144272, -0.0024898844, -0.010364867, -0.0021460776, -0.006572136, 0.00932259, -0.00020922447, 0.042530697, 0.03729036, 0.011291335, 0.011168289, -0.0085336445, 0.008063172, -0.002752263, 0.011696666, 0.02505808, 0.0033548295, -0.016531674, 0.0023740758, 0.0068109916, 0.017602902, -0.0028934048, 0.038448445, 0.021945724, 0.009756872, -0.015156447, 0.019586125, -0.0005460193, 0.0009762301, 0.005511764, 0.022018105, 0.0044369157, 0.025651598, 0.021453537, 0.024913318, 0.026071405, -0.016719863, 0.0021551251, -0.022133913, -0.00021487918, -0.017544998, 0.03320811, 0.013578555, 0.008374407, -0.004212537, -0.0056275725, 0.021120587, 0.019397935, 0.01683567, -0.0008278503, -0.0002687121, 0.0039664437, 0.031355172, -0.018225374, -0.0031195935, -0.039635483, 0.01533016, -0.0073864153, -0.0049652923, -0.020121738, -0.014729403, 0.011190003, 0.036305986, 0.011747332, -0.0053633843, 0.016589578, 0.008461264, 0.007498605, -0.033092298, -0.029733852, 0.03147098, 0.009394971, -0.016604055, -0.004932721, -0.0098437285, 0.0042523458, -0.017950328, -0.0016448436, -0.02649121, 0.008287551, -0.0001472488, 0.0022039819, 0.002207601, 0.015431492, 0.04012767, -0.0073972726, -0.010415534, -0.015040638, -0.003640732, -0.019021558, -0.017125193, -0.026867589, 0.037608832, -8.640405e-05, -0.0022347434, -0.043051835, 0.0008807785, -0.03393191, 0.0067856587, -0.013600269, 0.0071728933, 0.0066879448, 0.034279335, 0.015909202, -0.014070742, 0.0059062373, 0.031181458, 0.015547301, -0.01462807, -0.018601751, -0.007976315, 0.033902958, 0.03769569, 0.008381645, 0.008005268, 0.006098045, -0.01702386, 0.021800963, -0.0007133989, -0.03474257, 0.010430009, 0.008692881, 0.004263203, -0.014049027, -0.0034362574, -0.02391447, 0.011066956, 0.0031340695, 0.0062174727, -0.02701235, 0.025738455, -0.018833369, -0.0021804583, 0.01971641, 0.014678736, 0.03511895, 0.022452386, 0.027605869, 0.01647377, -0.0024735988, 0.010140488, -0.026621494, 0.006087188, 0.019774314, -0.0026708352, 0.043138694, 0.008815927, 0.010256297, -0.009351542, -0.0004252346, 0.0044513918, 0.019846695, 0.015315684, 0.010039155, 0.022655051, -0.016242152, 0.024898842, -0.0060184267, -0.010466199, 0.011479525, 0.013426556, 0.0060148076, -0.0015941773, -0.0033204488, 0.0075492715, 0.0032263545, 0.026360925, 0.009394971, -0.0266794, -0.0026255974, -0.010444486, -0.014049027, -0.025434457, 0.007527557, 0.015706537, -0.021468014, -0.0022618861, -0.0034760667, -0.020498117, -0.02486989, -0.0051498623, -0.0048024366, -0.0075492715, -0.016965955, -0.05040568, -0.0073864153, 0.012188852, 0.019224223, 0.013100845, 0.010683341, -0.00836717, 0.010017442, -0.021496966, -0.0371456, 0.0071620364, -0.022408959, -0.019890122, 0.035495326, -0.0011309431, -0.021844393, -0.034163527, 0.017009383, -0.014685974, 0.0014720355, 0.040648807, -0.021670679, 0.01886232, 0.008996879, 0.009163354, 0.016994908, 0.0017868901, -0.024131611, -0.008808689, 0.004628724, -0.00294769, -0.005243957, -0.003334925, 0.02727292, -0.009351542, 0.011291335, 0.0075782235, -0.024363229, 0.02100478, -0.022032581, 0.0013535126, 0.00015165946, 0.021395633, 0.0134337945, 0.019832218, 0.018616227, -0.012840276, 0.011428858, 0.014251692, -0.028749477, 0.0064888992, 0.011066956, -0.033989817, 0.0038072069, 0.0031738786, -0.03323706, 0.003257116, -6.480304e-05, -0.005207767, 0.030168133, -0.0016041297, -0.01809509, -0.042878125, -0.0071656555, 0.0064237565, -0.0026780732, -0.025550267, -0.008374407, 0.012101996, 0.017154144, 0.0066408976, 0.0022347434, -0.012427707, -0.02660702, 0.0058591897, 0.018138517, -0.015026162, 0.017081764, 0.0059569036, 0.0058121425, -0.009308114, 0.005486431, -0.0151998745, -0.0066408976, 0.0010513247, -0.007234417, 0.02405923, 0.019470315, 0.020469164, -0.0070136567, -0.0056022396, 0.0035574946, -0.012709991, -0.009713444, -0.02594112, 0.0006003046, 0.00010263309, 0.006253663, 0.001639415, 0.008497454, 0.008063172, -0.014201026, 0.00681823, 0.0018818893, -0.033845056, -0.019079462, -0.009242971, -0.01897813, 0.010914958, 0.0011517524, -0.015822345, -0.0008151838, -0.034684666, -0.0045744386, 0.015938155, 0.036537603, 0.012724467, -0.004491201, 0.025984548, -0.0036678745, 0.017400239, 0.02074421, 0.020975828, -0.028735003, 0.0102997245, -0.009778586, -0.014244454, 0.013289033, 0.0055841445, -0.0052186237, 0.011819712, -0.015605205, -0.010458961, -0.0075637475, -9.115401e-05, 0.0056022396, 0.021439062, 0.007983553, 0.0029874993, -0.02465275, -0.022365531, -0.017472617, 0.01050239, 0.01683567, 0.00530548, 0.021192968, 0.0056311917, -0.01783452, 0.01569206, -0.022336578, 0.015055114, 0.020136215, -0.00064689945, 0.00508472, -0.006894229, -0.004473106, -0.041864797, 0.010097059, -0.002108078, 0.007122227, 0.022828765, 0.012941608, -0.030283941, -0.006307948, 0.030052325, -0.040040813, -0.010972862, 0.028981095, 0.00202665, 0.009329828, -0.0047409134, -0.002620169, -0.031818405, 0.021323252, -0.019441364, -0.0007233512, 0.00637309, -0.012485611, -0.0021116969, 0.022191817, -0.025564741, 0.013802934, 0.01661853, -0.0080486955, -0.008678405, -0.020990303, 0.012319136, 0.001820366, 0.0043790117, -0.004227013, 0.028923191, 0.018051662, 0.002019412, -0.0008210647, -0.005298242, -0.007324892, 0.028749477, 0.029169284, -0.007679556, -0.029314045, 0.022712955, -0.0133758895, -0.005652906, -0.0143168345, -0.022423435, -0.00508472, -0.013361414, -0.0107557215, 0.0007848745, 0.016676433, -0.0039736815, -0.015503872, -0.012760657, -0.025680551, -0.0121816145, 0.0039121583, 0.010328677, -0.034076672, -0.0236539, 0.013658173, 0.0030454036, 0.025955597, -0.008888308, 0.0015833203, 0.0006437328, 0.005291004, -0.02358152, 0.01182695, 0.0066589927, 0.004675771, -0.01547492, 0.02181544, -0.025347602, -0.0040822523, 0.008924498, 0.011935521, -0.010183916, 0.002243791, 0.01750157, 0.0109801, 0.001738938, -0.022191817, 0.008200695, -0.0018855083, 0.025043603, -0.005649287, 0.0001686236, -0.011544667, -0.0024953128, 0.021279825, 0.015793394, -0.034829427, 0.0012150853, 0.0026708352, -0.00030105704, -0.0147438785, -0.04765523, -0.0074841287, 0.0074841287, 0.032918587, -0.0074407007, -0.02682416, -0.011501239, 0.015185399, -0.010582008, 0.017993758, -0.0007871364, -0.0008206123, -0.011624285, -0.010495151, -0.016097391, 0.019557172, -0.001315513, 0.011356478, -0.019629553, -0.0007550176, -0.021945724, -0.007896697, -0.01547492, 0.016372437, 0.0016167962, -0.03807207, -0.019586125, -0.018428039, -0.025984548, -0.008099362, -0.017964805, 0.016994908, -0.0039193965, -0.008866594, 0.003577399, 0.025289698, 0.025014652, 0.011320288, -0.025955597, 0.015619681, -0.02362495, -0.020483641, 0.017544998, -0.02987861, -0.059091322, -0.0075709852, 0.034800477, -0.006901467, 0.014584642, 0.0017905091, -0.009004116, -0.008794214, 0.013882552, 0.014056265, 0.014135884, 0.012478374, 0.006271758, -0.03280278, 0.009228496, 0.034800477, 0.0013091797, -0.01783452, -0.0059062373, 0.027316347, -0.01186314, -0.0023306475, -0.02067183, -0.011334764, 0.0021533156, 0.0018375564, 0.0020899829, 0.015518349, -0.00039560389, 0.01425893, 0.0071728933, -0.004255965, -0.0089027835, 0.006083569, 0.00386873, -0.012586944, 0.0018674132, -0.017689759, -0.011703904, -0.030747175, -0.013875314, 0.024811987, -0.016575102, -0.0062174727, 0.002023031, 0.0010992767, -0.0013254653, 0.010343153, 0.0005623049, 0.0072706067, -0.0008843975, 0.028387576, -0.005888142, 0.0028735, 0.004241489, -0.014410929, -0.001555273, 0.0020809353, -0.048350077, -0.023972373, -0.009706206, -0.00021239111, 0.00807041, 0.0098726805, 0.19432679, -0.02288667, -0.0047119614, 0.007093275, -0.019687457, 0.001470226, 0.0030417845, 0.010857053, -0.006868896, 0.012724467, -0.011906569, -0.0022456006, -0.006506994, -0.00079844584, 0.014910353, -0.01993355, -0.038651112, -0.017487094, -0.044731062, 0.008171742, 0.025579218, -0.0025586456, -0.016111868, 0.0075926995, 0.018703084, 0.01134924, -0.010111536, -0.0033874006, 0.033323918, 0.0133469375, -0.005330813, -0.012717228, 0.0023451236, 0.00891726, 0.00021069469, -0.011139337, 0.007961839, -0.011899331, -0.00085182633, -0.021395633, -0.0009662778, 0.0074045104, -0.023711804, -0.025506837, 0.0015987011, 0.0042668222, 0.004831389, -0.033700295, 0.022973524, 0.008606024, -0.01588025, 0.0049761496, 0.027041301, 0.010545818, 0.02472513, -0.013600269, -0.0112117175, -0.01850042, -0.011646, 0.012738943, -0.017776616, 0.018760988, -0.015185399, 0.0063043293, -0.021453537, 0.004031586, -0.02851786, -0.011653237, 0.029053476, -0.01809509, 0.02682416, -0.012377041, 0.005873666, 0.024696177, -0.016748814, -0.0066879448, 0.016531674, 0.04609181, 0.019108415, 0.017052813, 0.010466199, -0.0015172733, -0.021453537, -0.019470315, -0.017559474, -0.025448933, 0.01378122, 0.004183585, -0.015706537, -0.011783523, 0.012217804, -0.012956084, -0.01934003, -0.025564741, 0.014389215, 0.007722984, -0.00045961526, 0.008273074, -0.019397935, -0.009945061, -0.0043790117, 0.049218643, 0.0016891765, 0.007462415, -0.024754083, -0.0024591226, 0.004856722, 0.00088982604, 0.011146575, -0.013383128, -0.015634157, -0.054111555, -0.021714106, -0.020541545, -0.0061559496, 0.0074117486, 0.009749634, -0.019759838, 0.00983649, -0.025014652, -0.0067965155, -0.020729734, 0.010914958, -0.00729594, 0.0006351376, 0.01219609, -0.021019256, -0.015923679, -0.021120587, -0.02682416, 0.0053814794, 0.007831555, -0.02332095, -0.011566381, -0.025173888, 0.016908051, -0.0116098095, -0.03419248, -0.018949177, 0.007853269, 0.017067289, 0.002730549, 0.022930097, -0.0057252864, -0.003984539, 0.004288536, 0.038043115, -0.010951148, -0.024768557, -0.0043319645, -0.016604055, -0.011812475, -0.014012837, -0.023784185, 0.021468014, 0.0018692227, -0.023639424, -0.005685477, 0.012717228, -0.009561446, -0.051419005, 0.007896697, 0.03879587, 0.005519002, -0.015040638, -0.021496966, -0.18876797, 0.028228339, 0.021989152, -0.022539243, 0.03228164, 0.018066138, -0.011769046, 0.010249059, -0.01588025, 0.008837641, -0.0025785503, -0.00026758114, -0.013122559, -0.009945061, -0.0109801, -0.006130616, -0.00803422, -0.0005926142, 0.058975514, 0.01061096, 0.008729071, -0.02081659, 0.0057687145, 0.0066264216, -0.012297423, -0.0027775962, -0.020324403, 0.005497288, -0.010212868, -0.018529372, 0.005211386, 0.013397604, 0.017313382, -0.008844879, -0.011016291, -0.012282946, -0.01363646, -0.02369733, -0.0081355525, 0.009539731, 0.020295452, 0.01938346, -0.007838793, -0.01742919, 0.009850967, 0.052345473, 0.035234757, -0.007067942, 0.00015007614, -0.017414713, 0.009612111, -0.012253994, -0.013477222, 0.0042089177, 0.003857873, -0.008403359, 0.00056094775, 0.0027251204, -0.013100845, -0.0023668376, 0.015749965, -0.015576253, 0.015634157, 0.014584642, -0.0047915797, -0.00935878, -0.0033892102, 0.013368652, -0.0041582515, 0.019701933, -0.0074189864, 0.010864291, -0.0017063669, -0.012261232, 9.408032e-06, 0.013824648, -0.02295905, 0.016155295, 0.009127163, -0.025839787, -0.019615076, 0.019846695, -0.01676329, 0.013129797, -0.006434614, 0.007230798, 0.0017344143, 0.0030019754, 0.00061342353, -0.009474589, 0.017212048, -0.02358152, -0.013173225, 0.0013100845, 0.01142162, 0.0055370973, -0.0098147765, 0.012833037, 0.016661959, -0.0055624302, 0.0016240343, 0.01588025, -0.0045708194, 0.0018266992, 0.028358623, -0.00018796274, -0.028083578, -0.006876134, 0.02586874, -0.014049027, -0.027649296, 0.019282127, 0.015909202, -0.0014991781, -0.010350391, 0.028488908, 0.008454026, -0.03213688, -0.0036099702, -0.037493024, 0.04418097, 0.004241489, -0.035842754, 0.016821194, -0.019948026, -0.0012250375, -0.11974606, -0.0022799813, 0.017241001, 0.008555358, 0.0030526416, 0.00013469532, 0.01219609, 0.046526093, -0.011016291, 0.008121076, -0.0126086585, -0.04971083, -0.010516866, -0.025304172, 0.03216583, 0.0063875667, -0.011117623, -0.0033258775, -0.019745361, 0.024681702, 0.022828765, -0.008786975, 0.016126344, -0.02362495, -0.016575102, -0.01897813, -0.0073321303, -0.0036714936, 0.016719863, -0.0022474101, -0.008417835, -0.039664436, 0.006307948, -0.005059387, 0.018442515, -0.017776616, -0.017733188, -4.2099357e-05, 0.02155487, -0.017414713, 0.009496303, 0.0027160728, -0.001665653, -0.051129483, -0.003345782, -0.026795208, -0.0066408976, 0.012485611, 0.019991454, -0.014548452, -0.032223735, -0.011052481, -0.019122891, -0.010415534, 0.04542591, -0.01716862, 0.007136703, 0.018181946, -0.0036172085, 0.0049073882, -0.006506994, 0.005457479, -0.008302027, 0.010914958, 0.019224223, -0.0031901642, 0.003302354, -0.026737304, 0.022611624, -0.0032752112, -0.025738455, 0.014070742, -0.018456992, 0.0070715607, -0.012405993, -0.010010203, -0.031094601, -0.01201514, -0.0020773162, -0.00074913667, -0.016908051, -0.025695026, -0.016850147, -0.009221258, -0.002951309, 0.02830072, 0.011551905, 0.0116098095, 0.016097391, -0.022466863, -0.0045563434, 0.020946875, -0.005703572, -0.041140996, -0.015894726, 0.026853113, 0.012941608, 0.000611614, -0.005678239, 0.0075999373, -0.01492483, 0.012044092, -0.06392633, 0.022930097, 2.8669407e-05, 0.009981251, 0.0013688935, 0.0004489844, 0.0057976665, 0.018268801, 0.003984539, 0.008294789, -0.044962678, 0.0055913827, -0.014172073, -0.036450747, -0.010596484, -0.0048060557, 0.010806387, -0.007534795, 0.018587276, 0.004780723, -0.007722984, -0.0058121425, -0.014468834, -0.020252025, 0.0073900344, -0.008251361, -0.010524104, 0.0025875978, -0.01964403, 0.007122227, 0.0070100375, -0.0010440866, -0.0038180638, 0.0445863, -0.02081659, 0.005714429, 0.0076506035, -0.006141473, 0.030283941, 0.0020990304, -0.015634157, 0.0018565562, 0.028807381, -0.039259106, 0.0051679574, -0.003530352, -0.002936833, 0.021598298, 0.008265837, -0.024203992, 0.02214839, 0.031239362, 0.007060704, -0.033323918, -0.0053344322, -0.027041301, -0.01952822, 0.0065214704, -0.0014874163, -0.01934003, 0.052200712, -0.014237217, 0.029328521, -0.0024971224, -0.009670015, 0.00961935, -0.013086368, 0.021062683, -0.011689427, -0.036421794, -0.0039266343, 0.0008798737, -0.0064237565, 0.017487094, 0.020700783, -0.0059532844, -0.009807538, 0.025115984, 0.008302027, 0.020223072, -0.01422274, -0.032223735, -0.024479037, -0.012442184, 0.037579883, -0.029314045, 0.0025767407, -0.010639912, -0.037174553, 0.024754083, -0.0048350077, 0.0044984394, -0.028459957, 0.009648302, -0.028170435, 0.0041292994, 0.0026075023, -0.00770127, -0.012999512, -0.0023994087, -0.006662612, 0.008598787, 0.004889293, -0.039317008, -0.023451235, 0.0015462254, -0.022655051, -0.04617867, 0.0005315432, 0.020628402, 0.02214839, 0.010111536, 0.01533016, 0.010068107, 0.007976315, 0.015894726, -0.00943116, -0.022264197, -0.029487757, 0.026621494, 0.002851786, 0.029342996, 0.038187876, 0.0014249883, 0.008316503, 0.016879098, 0.011370954, -0.017530523, 0.009662778, -0.009192306, 0.018630704, -0.013904267, -0.014244454, -0.007889459, -0.017805567, -0.018326707, 0.015040638, 0.014642546, -0.00022901598, 0.07753384, 0.019962503, -0.0039628246, 0.0072380356, 0.012413232, 0.011515715, 0.010154964, 0.0037094932, 0.0085625965, -0.026954444, 0.014765593, 0.010191155, -0.0016240343, -0.008729071, -0.0022890288, -0.0005347099, -0.0014448928, 0.008714595, -0.013643697, 0.025043603, 0.014266169, 0.00611614, -0.0024555037, 0.014403691, -0.00814279, -0.013839125, 0.036016464, 0.0032136878, 0.0019253175, -0.025564741, -0.0008088505, -0.008714595, -0.032918587, -0.025159413, 0.009561446, -0.013165987, -0.004657676, -0.022408959, 0.018949177, 0.016777767, 0.0080486955, 0.022466863, -0.032194782, -0.01831223, 0.003416353, -0.0023595996, -0.010864291, 0.007932887, -0.024565892],
    n_results=2
    )

print(results)
print(results["metadatas"])
print(results["ids"])

print("Top 3 results:")
for i, result in enumerate(results["documents"]):
    for j, chunk in enumerate(result):  # Iterate over each chunk in the result
        print(f"Chunk {j + 1}:")  # Print "Chunk 1", "Chunk 2", etc.
        print("*" * 50)  # Print a row of asterisks before each chunk
        print(chunk)  # Print the chunk on a separate line
        print("*" * 50)  # Print a row of asterisks after each chunk
        print("\n")


