__author__ = 'JF Chamberland'

import ccsfg as ccsfg
import numpy as np

class WIMAX_768_640(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 41, 83, 144, 162, 255, 285, 291, 349, 370, 412, 428, 450, 481, 525, 551, 578, 634, 667, 673])
        self.__Check2VarEdges.append([2, 42, 84, 145, 163, 256, 286, 292, 350, 371, 413, 429, 451, 482, 526, 552, 579, 635, 668, 674])
        self.__Check2VarEdges.append([3, 43, 85, 146, 164, 225, 287, 293, 351, 372, 414, 430, 452, 483, 527, 553, 580, 636, 669, 675])
        self.__Check2VarEdges.append([4, 44, 86, 147, 165, 226, 288, 294, 352, 373, 415, 431, 453, 484, 528, 554, 581, 637, 670, 676])
        self.__Check2VarEdges.append([5, 45, 87, 148, 166, 227, 257, 295, 321, 374, 416, 432, 454, 485, 529, 555, 582, 638, 671, 677])
        self.__Check2VarEdges.append([6, 46, 88, 149, 167, 228, 258, 296, 322, 375, 385, 433, 455, 486, 530, 556, 583, 639, 672, 678])
        self.__Check2VarEdges.append([7, 47, 89, 150, 168, 229, 259, 297, 323, 376, 386, 434, 456, 487, 531, 557, 584, 640, 641, 679])
        self.__Check2VarEdges.append([8, 48, 90, 151, 169, 230, 260, 298, 324, 377, 387, 435, 457, 488, 532, 558, 585, 609, 642, 680])
        self.__Check2VarEdges.append([9, 49, 91, 152, 170, 231, 261, 299, 325, 378, 388, 436, 458, 489, 533, 559, 586, 610, 643, 681])
        self.__Check2VarEdges.append([10, 50, 92, 153, 171, 232, 262, 300, 326, 379, 389, 437, 459, 490, 534, 560, 587, 611, 644, 682])
        self.__Check2VarEdges.append([11, 51, 93, 154, 172, 233, 263, 301, 327, 380, 390, 438, 460, 491, 535, 561, 588, 612, 645, 683])
        self.__Check2VarEdges.append([12, 52, 94, 155, 173, 234, 264, 302, 328, 381, 391, 439, 461, 492, 536, 562, 589, 613, 646, 684])
        self.__Check2VarEdges.append([13, 53, 95, 156, 174, 235, 265, 303, 329, 382, 392, 440, 462, 493, 537, 563, 590, 614, 647, 685])
        self.__Check2VarEdges.append([14, 54, 96, 157, 175, 236, 266, 304, 330, 383, 393, 441, 463, 494, 538, 564, 591, 615, 648, 686])
        self.__Check2VarEdges.append([15, 55, 65, 158, 176, 237, 267, 305, 331, 384, 394, 442, 464, 495, 539, 565, 592, 616, 649, 687])
        self.__Check2VarEdges.append([16, 56, 66, 159, 177, 238, 268, 306, 332, 353, 395, 443, 465, 496, 540, 566, 593, 617, 650, 688])
        self.__Check2VarEdges.append([17, 57, 67, 160, 178, 239, 269, 307, 333, 354, 396, 444, 466, 497, 541, 567, 594, 618, 651, 689])
        self.__Check2VarEdges.append([18, 58, 68, 129, 179, 240, 270, 308, 334, 355, 397, 445, 467, 498, 542, 568, 595, 619, 652, 690])
        self.__Check2VarEdges.append([19, 59, 69, 130, 180, 241, 271, 309, 335, 356, 398, 446, 468, 499, 543, 569, 596, 620, 653, 691])
        self.__Check2VarEdges.append([20, 60, 70, 131, 181, 242, 272, 310, 336, 357, 399, 447, 469, 500, 544, 570, 597, 621, 654, 692])
        self.__Check2VarEdges.append([21, 61, 71, 132, 182, 243, 273, 311, 337, 358, 400, 448, 470, 501, 513, 571, 598, 622, 655, 693])
        self.__Check2VarEdges.append([22, 62, 72, 133, 183, 244, 274, 312, 338, 359, 401, 417, 471, 502, 514, 572, 599, 623, 656, 694])
        self.__Check2VarEdges.append([23, 63, 73, 134, 184, 245, 275, 313, 339, 360, 402, 418, 472, 503, 515, 573, 600, 624, 657, 695])
        self.__Check2VarEdges.append([24, 64, 74, 135, 185, 246, 276, 314, 340, 361, 403, 419, 473, 504, 516, 574, 601, 625, 658, 696])
        self.__Check2VarEdges.append([25, 33, 75, 136, 186, 247, 277, 315, 341, 362, 404, 420, 474, 505, 517, 575, 602, 626, 659, 697])
        self.__Check2VarEdges.append([26, 34, 76, 137, 187, 248, 278, 316, 342, 363, 405, 421, 475, 506, 518, 576, 603, 627, 660, 698])
        self.__Check2VarEdges.append([27, 35, 77, 138, 188, 249, 279, 317, 343, 364, 406, 422, 476, 507, 519, 545, 604, 628, 661, 699])
        self.__Check2VarEdges.append([28, 36, 78, 139, 189, 250, 280, 318, 344, 365, 407, 423, 477, 508, 520, 546, 605, 629, 662, 700])
        self.__Check2VarEdges.append([29, 37, 79, 140, 190, 251, 281, 319, 345, 366, 408, 424, 478, 509, 521, 547, 606, 630, 663, 701])
        self.__Check2VarEdges.append([30, 38, 80, 141, 191, 252, 282, 320, 346, 367, 409, 425, 479, 510, 522, 548, 607, 631, 664, 702])
        self.__Check2VarEdges.append([31, 39, 81, 142, 192, 253, 283, 289, 347, 368, 410, 426, 480, 511, 523, 549, 608, 632, 665, 703])
        self.__Check2VarEdges.append([32, 40, 82, 143, 161, 254, 284, 290, 348, 369, 411, 427, 449, 512, 524, 550, 577, 633, 666, 704])
        self.__Check2VarEdges.append([35, 109, 142, 176, 197, 251, 272, 334, 360, 389, 440, 453, 505, 513, 559, 593, 609, 641, 673, 705])
        self.__Check2VarEdges.append([36, 110, 143, 177, 198, 252, 273, 335, 361, 390, 441, 454, 506, 514, 560, 594, 610, 642, 674, 706])
        self.__Check2VarEdges.append([37, 111, 144, 178, 199, 253, 274, 336, 362, 391, 442, 455, 507, 515, 561, 595, 611, 643, 675, 707])
        self.__Check2VarEdges.append([38, 112, 145, 179, 200, 254, 275, 337, 363, 392, 443, 456, 508, 516, 562, 596, 612, 644, 676, 708])
        self.__Check2VarEdges.append([39, 113, 146, 180, 201, 255, 276, 338, 364, 393, 444, 457, 509, 517, 563, 597, 613, 645, 677, 709])
        self.__Check2VarEdges.append([40, 114, 147, 181, 202, 256, 277, 339, 365, 394, 445, 458, 510, 518, 564, 598, 614, 646, 678, 710])
        self.__Check2VarEdges.append([41, 115, 148, 182, 203, 225, 278, 340, 366, 395, 446, 459, 511, 519, 565, 599, 615, 647, 679, 711])
        self.__Check2VarEdges.append([42, 116, 149, 183, 204, 226, 279, 341, 367, 396, 447, 460, 512, 520, 566, 600, 616, 648, 680, 712])
        self.__Check2VarEdges.append([43, 117, 150, 184, 205, 227, 280, 342, 368, 397, 448, 461, 481, 521, 567, 601, 617, 649, 681, 713])
        self.__Check2VarEdges.append([44, 118, 151, 185, 206, 228, 281, 343, 369, 398, 417, 462, 482, 522, 568, 602, 618, 650, 682, 714])
        self.__Check2VarEdges.append([45, 119, 152, 186, 207, 229, 282, 344, 370, 399, 418, 463, 483, 523, 569, 603, 619, 651, 683, 715])
        self.__Check2VarEdges.append([46, 120, 153, 187, 208, 230, 283, 345, 371, 400, 419, 464, 484, 524, 570, 604, 620, 652, 684, 716])
        self.__Check2VarEdges.append([47, 121, 154, 188, 209, 231, 284, 346, 372, 401, 420, 465, 485, 525, 571, 605, 621, 653, 685, 717])
        self.__Check2VarEdges.append([48, 122, 155, 189, 210, 232, 285, 347, 373, 402, 421, 466, 486, 526, 572, 606, 622, 654, 686, 718])
        self.__Check2VarEdges.append([49, 123, 156, 190, 211, 233, 286, 348, 374, 403, 422, 467, 487, 527, 573, 607, 623, 655, 687, 719])
        self.__Check2VarEdges.append([50, 124, 157, 191, 212, 234, 287, 349, 375, 404, 423, 468, 488, 528, 574, 608, 624, 656, 688, 720])
        self.__Check2VarEdges.append([51, 125, 158, 192, 213, 235, 288, 350, 376, 405, 424, 469, 489, 529, 575, 577, 625, 657, 689, 721])
        self.__Check2VarEdges.append([52, 126, 159, 161, 214, 236, 257, 351, 377, 406, 425, 470, 490, 530, 576, 578, 626, 658, 690, 722])
        self.__Check2VarEdges.append([53, 127, 160, 162, 215, 237, 258, 352, 378, 407, 426, 471, 491, 531, 545, 579, 627, 659, 691, 723])
        self.__Check2VarEdges.append([54, 128, 129, 163, 216, 238, 259, 321, 379, 408, 427, 472, 492, 532, 546, 580, 628, 660, 692, 724])
        self.__Check2VarEdges.append([55, 97, 130, 164, 217, 239, 260, 322, 380, 409, 428, 473, 493, 533, 547, 581, 629, 661, 693, 725])
        self.__Check2VarEdges.append([56, 98, 131, 165, 218, 240, 261, 323, 381, 410, 429, 474, 494, 534, 548, 582, 630, 662, 694, 726])
        self.__Check2VarEdges.append([57, 99, 132, 166, 219, 241, 262, 324, 382, 411, 430, 475, 495, 535, 549, 583, 631, 663, 695, 727])
        self.__Check2VarEdges.append([58, 100, 133, 167, 220, 242, 263, 325, 383, 412, 431, 476, 496, 536, 550, 584, 632, 664, 696, 728])
        self.__Check2VarEdges.append([59, 101, 134, 168, 221, 243, 264, 326, 384, 413, 432, 477, 497, 537, 551, 585, 633, 665, 697, 729])
        self.__Check2VarEdges.append([60, 102, 135, 169, 222, 244, 265, 327, 353, 414, 433, 478, 498, 538, 552, 586, 634, 666, 698, 730])
        self.__Check2VarEdges.append([61, 103, 136, 170, 223, 245, 266, 328, 354, 415, 434, 479, 499, 539, 553, 587, 635, 667, 699, 731])
        self.__Check2VarEdges.append([62, 104, 137, 171, 224, 246, 267, 329, 355, 416, 435, 480, 500, 540, 554, 588, 636, 668, 700, 732])
        self.__Check2VarEdges.append([63, 105, 138, 172, 193, 247, 268, 330, 356, 385, 436, 449, 501, 541, 555, 589, 637, 669, 701, 733])
        self.__Check2VarEdges.append([64, 106, 139, 173, 194, 248, 269, 331, 357, 386, 437, 450, 502, 542, 556, 590, 638, 670, 702, 734])
        self.__Check2VarEdges.append([33, 107, 140, 174, 195, 249, 270, 332, 358, 387, 438, 451, 503, 543, 557, 591, 639, 671, 703, 735])
        self.__Check2VarEdges.append([34, 108, 141, 175, 196, 250, 271, 333, 359, 388, 439, 452, 504, 544, 558, 592, 640, 672, 704, 736])
        self.__Check2VarEdges.append([18, 60, 92, 98, 151, 200, 267, 297, 351, 373, 412, 420, 477, 507, 533, 574, 599, 614, 705, 737])
        self.__Check2VarEdges.append([19, 61, 93, 99, 152, 201, 268, 298, 352, 374, 413, 421, 478, 508, 534, 575, 600, 615, 706, 738])
        self.__Check2VarEdges.append([20, 62, 94, 100, 153, 202, 269, 299, 321, 375, 414, 422, 479, 509, 535, 576, 601, 616, 707, 739])
        self.__Check2VarEdges.append([21, 63, 95, 101, 154, 203, 270, 300, 322, 376, 415, 423, 480, 510, 536, 545, 602, 617, 708, 740])
        self.__Check2VarEdges.append([22, 64, 96, 102, 155, 204, 271, 301, 323, 377, 416, 424, 449, 511, 537, 546, 603, 618, 709, 741])
        self.__Check2VarEdges.append([23, 33, 65, 103, 156, 205, 272, 302, 324, 378, 385, 425, 450, 512, 538, 547, 604, 619, 710, 742])
        self.__Check2VarEdges.append([24, 34, 66, 104, 157, 206, 273, 303, 325, 379, 386, 426, 451, 481, 539, 548, 605, 620, 711, 743])
        self.__Check2VarEdges.append([25, 35, 67, 105, 158, 207, 274, 304, 326, 380, 387, 427, 452, 482, 540, 549, 606, 621, 712, 744])
        self.__Check2VarEdges.append([26, 36, 68, 106, 159, 208, 275, 305, 327, 381, 388, 428, 453, 483, 541, 550, 607, 622, 713, 745])
        self.__Check2VarEdges.append([27, 37, 69, 107, 160, 209, 276, 306, 328, 382, 389, 429, 454, 484, 542, 551, 608, 623, 714, 746])
        self.__Check2VarEdges.append([28, 38, 70, 108, 129, 210, 277, 307, 329, 383, 390, 430, 455, 485, 543, 552, 577, 624, 715, 747])
        self.__Check2VarEdges.append([29, 39, 71, 109, 130, 211, 278, 308, 330, 384, 391, 431, 456, 486, 544, 553, 578, 625, 716, 748])
        self.__Check2VarEdges.append([30, 40, 72, 110, 131, 212, 279, 309, 331, 353, 392, 432, 457, 487, 513, 554, 579, 626, 717, 749])
        self.__Check2VarEdges.append([31, 41, 73, 111, 132, 213, 280, 310, 332, 354, 393, 433, 458, 488, 514, 555, 580, 627, 718, 750])
        self.__Check2VarEdges.append([32, 42, 74, 112, 133, 214, 281, 311, 333, 355, 394, 434, 459, 489, 515, 556, 581, 628, 719, 751])
        self.__Check2VarEdges.append([1, 43, 75, 113, 134, 215, 282, 312, 334, 356, 395, 435, 460, 490, 516, 557, 582, 629, 720, 752])
        self.__Check2VarEdges.append([2, 44, 76, 114, 135, 216, 283, 313, 335, 357, 396, 436, 461, 491, 517, 558, 583, 630, 721, 753])
        self.__Check2VarEdges.append([3, 45, 77, 115, 136, 217, 284, 314, 336, 358, 397, 437, 462, 492, 518, 559, 584, 631, 722, 754])
        self.__Check2VarEdges.append([4, 46, 78, 116, 137, 218, 285, 315, 337, 359, 398, 438, 463, 493, 519, 560, 585, 632, 723, 755])
        self.__Check2VarEdges.append([5, 47, 79, 117, 138, 219, 286, 316, 338, 360, 399, 439, 464, 494, 520, 561, 586, 633, 724, 756])
        self.__Check2VarEdges.append([6, 48, 80, 118, 139, 220, 287, 317, 339, 361, 400, 440, 465, 495, 521, 562, 587, 634, 725, 757])
        self.__Check2VarEdges.append([7, 49, 81, 119, 140, 221, 288, 318, 340, 362, 401, 441, 466, 496, 522, 563, 588, 635, 726, 758])
        self.__Check2VarEdges.append([8, 50, 82, 120, 141, 222, 257, 319, 341, 363, 402, 442, 467, 497, 523, 564, 589, 636, 727, 759])
        self.__Check2VarEdges.append([9, 51, 83, 121, 142, 223, 258, 320, 342, 364, 403, 443, 468, 498, 524, 565, 590, 637, 728, 760])
        self.__Check2VarEdges.append([10, 52, 84, 122, 143, 224, 259, 289, 343, 365, 404, 444, 469, 499, 525, 566, 591, 638, 729, 761])
        self.__Check2VarEdges.append([11, 53, 85, 123, 144, 193, 260, 290, 344, 366, 405, 445, 470, 500, 526, 567, 592, 639, 730, 762])
        self.__Check2VarEdges.append([12, 54, 86, 124, 145, 194, 261, 291, 345, 367, 406, 446, 471, 501, 527, 568, 593, 640, 731, 763])
        self.__Check2VarEdges.append([13, 55, 87, 125, 146, 195, 262, 292, 346, 368, 407, 447, 472, 502, 528, 569, 594, 609, 732, 764])
        self.__Check2VarEdges.append([14, 56, 88, 126, 147, 196, 263, 293, 347, 369, 408, 448, 473, 503, 529, 570, 595, 610, 733, 765])
        self.__Check2VarEdges.append([15, 57, 89, 127, 148, 197, 264, 294, 348, 370, 409, 417, 474, 504, 530, 571, 596, 611, 734, 766])
        self.__Check2VarEdges.append([16, 58, 90, 128, 149, 198, 265, 295, 349, 371, 410, 418, 475, 505, 531, 572, 597, 612, 735, 767])
        self.__Check2VarEdges.append([17, 59, 91, 97, 150, 199, 266, 296, 350, 372, 411, 419, 476, 506, 532, 573, 598, 613, 736, 768])
        self.__Check2VarEdges.append([17, 81, 102, 173, 197, 228, 260, 295, 338, 383, 394, 447, 468, 491, 541, 575, 580, 631, 667, 737])
        self.__Check2VarEdges.append([18, 82, 103, 174, 198, 229, 261, 296, 339, 384, 395, 448, 469, 492, 542, 576, 581, 632, 668, 738])
        self.__Check2VarEdges.append([19, 83, 104, 175, 199, 230, 262, 297, 340, 353, 396, 417, 470, 493, 543, 545, 582, 633, 669, 739])
        self.__Check2VarEdges.append([20, 84, 105, 176, 200, 231, 263, 298, 341, 354, 397, 418, 471, 494, 544, 546, 583, 634, 670, 740])
        self.__Check2VarEdges.append([21, 85, 106, 177, 201, 232, 264, 299, 342, 355, 398, 419, 472, 495, 513, 547, 584, 635, 671, 741])
        self.__Check2VarEdges.append([22, 86, 107, 178, 202, 233, 265, 300, 343, 356, 399, 420, 473, 496, 514, 548, 585, 636, 672, 742])
        self.__Check2VarEdges.append([23, 87, 108, 179, 203, 234, 266, 301, 344, 357, 400, 421, 474, 497, 515, 549, 586, 637, 641, 743])
        self.__Check2VarEdges.append([24, 88, 109, 180, 204, 235, 267, 302, 345, 358, 401, 422, 475, 498, 516, 550, 587, 638, 642, 744])
        self.__Check2VarEdges.append([25, 89, 110, 181, 205, 236, 268, 303, 346, 359, 402, 423, 476, 499, 517, 551, 588, 639, 643, 745])
        self.__Check2VarEdges.append([26, 90, 111, 182, 206, 237, 269, 304, 347, 360, 403, 424, 477, 500, 518, 552, 589, 640, 644, 746])
        self.__Check2VarEdges.append([27, 91, 112, 183, 207, 238, 270, 305, 348, 361, 404, 425, 478, 501, 519, 553, 590, 609, 645, 747])
        self.__Check2VarEdges.append([28, 92, 113, 184, 208, 239, 271, 306, 349, 362, 405, 426, 479, 502, 520, 554, 591, 610, 646, 748])
        self.__Check2VarEdges.append([29, 93, 114, 185, 209, 240, 272, 307, 350, 363, 406, 427, 480, 503, 521, 555, 592, 611, 647, 749])
        self.__Check2VarEdges.append([30, 94, 115, 186, 210, 241, 273, 308, 351, 364, 407, 428, 449, 504, 522, 556, 593, 612, 648, 750])
        self.__Check2VarEdges.append([31, 95, 116, 187, 211, 242, 274, 309, 352, 365, 408, 429, 450, 505, 523, 557, 594, 613, 649, 751])
        self.__Check2VarEdges.append([32, 96, 117, 188, 212, 243, 275, 310, 321, 366, 409, 430, 451, 506, 524, 558, 595, 614, 650, 752])
        self.__Check2VarEdges.append([1, 65, 118, 189, 213, 244, 276, 311, 322, 367, 410, 431, 452, 507, 525, 559, 596, 615, 651, 753])
        self.__Check2VarEdges.append([2, 66, 119, 190, 214, 245, 277, 312, 323, 368, 411, 432, 453, 508, 526, 560, 597, 616, 652, 754])
        self.__Check2VarEdges.append([3, 67, 120, 191, 215, 246, 278, 313, 324, 369, 412, 433, 454, 509, 527, 561, 598, 617, 653, 755])
        self.__Check2VarEdges.append([4, 68, 121, 192, 216, 247, 279, 314, 325, 370, 413, 434, 455, 510, 528, 562, 599, 618, 654, 756])
        self.__Check2VarEdges.append([5, 69, 122, 161, 217, 248, 280, 315, 326, 371, 414, 435, 456, 511, 529, 563, 600, 619, 655, 757])
        self.__Check2VarEdges.append([6, 70, 123, 162, 218, 249, 281, 316, 327, 372, 415, 436, 457, 512, 530, 564, 601, 620, 656, 758])
        self.__Check2VarEdges.append([7, 71, 124, 163, 219, 250, 282, 317, 328, 373, 416, 437, 458, 481, 531, 565, 602, 621, 657, 759])
        self.__Check2VarEdges.append([8, 72, 125, 164, 220, 251, 283, 318, 329, 374, 385, 438, 459, 482, 532, 566, 603, 622, 658, 760])
        self.__Check2VarEdges.append([9, 73, 126, 165, 221, 252, 284, 319, 330, 375, 386, 439, 460, 483, 533, 567, 604, 623, 659, 761])
        self.__Check2VarEdges.append([10, 74, 127, 166, 222, 253, 285, 320, 331, 376, 387, 440, 461, 484, 534, 568, 605, 624, 660, 762])
        self.__Check2VarEdges.append([11, 75, 128, 167, 223, 254, 286, 289, 332, 377, 388, 441, 462, 485, 535, 569, 606, 625, 661, 763])
        self.__Check2VarEdges.append([12, 76, 97, 168, 224, 255, 287, 290, 333, 378, 389, 442, 463, 486, 536, 570, 607, 626, 662, 764])
        self.__Check2VarEdges.append([13, 77, 98, 169, 193, 256, 288, 291, 334, 379, 390, 443, 464, 487, 537, 571, 608, 627, 663, 765])
        self.__Check2VarEdges.append([14, 78, 99, 170, 194, 225, 257, 292, 335, 380, 391, 444, 465, 488, 538, 572, 577, 628, 664, 766])
        self.__Check2VarEdges.append([15, 79, 100, 171, 195, 226, 258, 293, 336, 381, 392, 445, 466, 489, 539, 573, 578, 629, 665, 767])
        self.__Check2VarEdges.append([16, 80, 101, 172, 196, 227, 259, 294, 337, 382, 393, 446, 467, 490, 540, 574, 579, 630, 666, 768])
        self.__SystematicVars = list(range(1,640+1))
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DMIN = 7

    def getmaxdepth(self):
        return self.__DMIN


class Triadic6(ccsfg.Encoding):

    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 1])
        super().__init__(self.__Check2VarEdges, [1, 3, 5, 7, 9, 11], seclength)
        self.__DepthFromRoot = 6  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Triadic8(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 13])
        self.__Check2VarEdges.append([13, 14, 15])
        self.__Check2VarEdges.append([15, 16, 1])
        self.__SystematicVars = [1, 3, 5, 7, 9, 11, 13, 15]
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DepthFromRoot = 8

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Triadic10(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 13])
        self.__Check2VarEdges.append([13, 14, 15])
        self.__Check2VarEdges.append([15, 16, 17])
        self.__Check2VarEdges.append([17, 18, 19])
        self.__Check2VarEdges.append([19, 20, 1])
        self.__SystematicVars = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DepthFromRoot = 10

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Triadic12(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 13])
        self.__Check2VarEdges.append([13, 14, 15])
        self.__Check2VarEdges.append([15, 16, 17])
        self.__Check2VarEdges.append([17, 18, 19])
        self.__Check2VarEdges.append([19, 20, 21])
        self.__Check2VarEdges.append([21, 22, 23])
        self.__Check2VarEdges.append([23, 24, 1])
        self.__SystematicVars = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DepthFromRoot = 12

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Triadic15(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 13])
        self.__Check2VarEdges.append([13, 14, 15])
        self.__Check2VarEdges.append([15, 16, 17])
        self.__Check2VarEdges.append([17, 18, 19])
        self.__Check2VarEdges.append([19, 20, 21])
        self.__Check2VarEdges.append([21, 22, 23])
        self.__Check2VarEdges.append([23, 24, 25])
        self.__Check2VarEdges.append([25, 26, 27])
        self.__Check2VarEdges.append([27, 28, 29])
        self.__Check2VarEdges.append([29, 30, 1])
        self.__SystematicVars = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DepthFromRoot = 15

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Triadic16(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([3, 4, 5])
        self.__Check2VarEdges.append([5, 6, 7])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([9, 10, 11])
        self.__Check2VarEdges.append([11, 12, 13])
        self.__Check2VarEdges.append([13, 14, 15])
        self.__Check2VarEdges.append([15, 16, 17])
        self.__Check2VarEdges.append([17, 18, 19])
        self.__Check2VarEdges.append([19, 20, 21])
        self.__Check2VarEdges.append([21, 22, 23])
        self.__Check2VarEdges.append([23, 24, 25])
        self.__Check2VarEdges.append([25, 26, 27])
        self.__Check2VarEdges.append([27, 28, 29])
        self.__Check2VarEdges.append([29, 30, 31])
        self.__Check2VarEdges.append([31, 32, 1])
        super().__init__(self.__Check2VarEdges, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], seclength)
        self.__DepthFromRoot = 16

    def getmaxdepth(self):
        return self.__DepthFromRoot


class CCSDS_ldpc_n32_k16(ccsfg.Encoding):
    def __init__(self, seclength=8):
        # https://www.uni-kl.de/fileadmin/chaco/public/alists_ccsds/CCSDS_ldpc_n32_k16.alist
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([3, 4, 6, 9, 15, 17, 24, 29])
        self.__Check2VarEdges.append([1, 4, 7, 10, 16, 18, 21, 30])
        self.__Check2VarEdges.append([1, 2, 8, 11, 13, 19, 22, 31])
        self.__Check2VarEdges.append([2, 3, 5, 12, 14, 20, 23, 32])
        self.__Check2VarEdges.append([1, 5, 6, 9, 13, 17, 21, 25])
        self.__Check2VarEdges.append([2, 6, 7, 10, 14, 18, 22, 26])
        self.__Check2VarEdges.append([3, 7, 8, 11, 15, 19, 23, 27])
        self.__Check2VarEdges.append([4, 5, 8, 12, 16, 20, 24, 28])
        self.__Check2VarEdges.append([4, 5, 9, 11, 13, 21, 26, 29])
        self.__Check2VarEdges.append([1, 6, 10, 12, 14, 22, 27, 30])
        self.__Check2VarEdges.append([2, 7, 9, 11, 15, 23, 28, 31])
        self.__Check2VarEdges.append([3, 8, 10, 12, 16, 24, 25, 32])
        self.__Check2VarEdges.append([3, 5, 9, 13, 16, 17, 25, 29])
        self.__Check2VarEdges.append([4, 6, 10, 13, 14, 18, 26, 30])
        self.__Check2VarEdges.append([1, 7, 11, 14, 15, 19, 27, 31])
        self.__Check2VarEdges.append([2, 8, 12, 15, 16, 20, 28, 32])
        super().__init__(self.__Check2VarEdges, None, seclength)
        self.__DepthFromRoot = 16  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot

class MacKay_96_33_964(ccsfg.Encoding):
    def __init__(self, seclength=8):
        # https://www.uni-kl.de/fileadmin/chaco/public/alists_misc/MacKay_96.33.964.alist
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([23, 96, 3, 64, 16, 90])
        self.__Check2VarEdges.append([12, 57, 19, 70, 38, 64])
        self.__Check2VarEdges.append([4, 95, 20, 52, 30, 70])
        self.__Check2VarEdges.append([42, 55, 1, 87, 31, 66])
        self.__Check2VarEdges.append([28, 87, 41, 77, 9, 69])
        self.__Check2VarEdges.append([43, 78, 29, 84, 22, 87])
        self.__Check2VarEdges.append([34, 49, 14, 93, 6, 92])
        self.__Check2VarEdges.append([41, 82, 24, 55, 32, 83])
        self.__Check2VarEdges.append([24, 66, 5, 73, 26, 58])
        self.__Check2VarEdges.append([44, 91, 33, 79, 12, 61])
        self.__Check2VarEdges.append([3, 72, 32, 67, 19, 63])
        self.__Check2VarEdges.append([22, 92, 25, 62, 24, 75])
        self.__Check2VarEdges.append([39, 93, 22, 94, 7, 60])
        self.__Check2VarEdges.append([48, 84, 26, 92, 28, 94])
        self.__Check2VarEdges.append([37, 70, 7, 85, 40, 96])
        self.__Check2VarEdges.append([15, 71, 17, 75, 46, 79])
        self.__Check2VarEdges.append([6, 64, 46, 69, 43, 54])
        self.__Check2VarEdges.append([14, 63, 11, 81, 44, 82])
        self.__Check2VarEdges.append([36, 68, 35, 54, 11, 52])
        self.__Check2VarEdges.append([45, 73, 36, 72, 39, 95])
        self.__Check2VarEdges.append([20, 54, 21, 71, 1, 50])
        self.__Check2VarEdges.append([9, 51, 6, 89, 17, 88])
        self.__Check2VarEdges.append([33, 74, 30, 50, 47, 78])
        self.__Check2VarEdges.append([32, 75, 16, 76, 29, 68])
        self.__Check2VarEdges.append([17, 50, 18, 91, 36, 84])
        self.__Check2VarEdges.append([16, 89, 42, 95, 27, 86])
        self.__Check2VarEdges.append([11, 88, 45, 90, 13, 53])
        self.__Check2VarEdges.append([10, 81, 8, 57, 21, 55])
        self.__Check2VarEdges.append([26, 85, 27, 68, 45, 65])
        self.__Check2VarEdges.append([46, 94, 31, 58, 10, 59])
        self.__Check2VarEdges.append([27, 79, 38, 49, 2, 56])
        self.__Check2VarEdges.append([35, 62, 15, 83, 14, 57])
        self.__Check2VarEdges.append([2, 61, 10, 80, 3, 71])
        self.__Check2VarEdges.append([21, 60, 12, 51, 18, 67])
        self.__Check2VarEdges.append([18, 56, 34, 53, 41, 80])
        self.__Check2VarEdges.append([29, 83, 47, 60, 5, 85])
        self.__Check2VarEdges.append([19, 58, 43, 96, 4, 76])
        self.__Check2VarEdges.append([13, 59, 2, 86, 23, 74])
        self.__Check2VarEdges.append([38, 77, 23, 65, 20, 91])
        self.__Check2VarEdges.append([8, 53, 40, 59, 48, 62])
        self.__Check2VarEdges.append([47, 76, 13, 63, 35, 51])
        self.__Check2VarEdges.append([5, 52, 9, 82, 33, 49])
        self.__Check2VarEdges.append([40, 90, 39, 78, 34, 77])
        self.__Check2VarEdges.append([25, 69, 48, 88, 42, 72])
        self.__Check2VarEdges.append([31, 65, 44, 56, 15, 89])
        self.__Check2VarEdges.append([30, 86, 28, 61, 37, 93])
        self.__Check2VarEdges.append([1, 67, 37, 74, 8, 73])
        self.__Check2VarEdges.append([7, 80, 4, 66, 25, 81])
        super().__init__(self.__Check2VarEdges, None, seclength)
        self.__DepthFromRoot = 24  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class WRAN_N384_K192(ccsfg.Encoding):
    def __init__(self, seclength=16):
        # https://www.uni-kl.de/fileadmin/chaco/public/alists_wran/WRAN_N384_K192_P16_R05.txt
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([32, 45, 138, 158, 194, 209])
        self.__Check2VarEdges.append([17, 46, 139, 159, 195, 210])
        self.__Check2VarEdges.append([18, 47, 140, 160, 196, 211])
        self.__Check2VarEdges.append([19, 48, 141, 145, 197, 212])
        self.__Check2VarEdges.append([20, 33, 142, 146, 198, 213])
        self.__Check2VarEdges.append([21, 34, 143, 147, 199, 214])
        self.__Check2VarEdges.append([22, 35, 144, 148, 200, 215])
        self.__Check2VarEdges.append([23, 36, 129, 149, 201, 216])
        self.__Check2VarEdges.append([24, 37, 130, 150, 202, 217])
        self.__Check2VarEdges.append([25, 38, 131, 151, 203, 218])
        self.__Check2VarEdges.append([26, 39, 132, 152, 204, 219])
        self.__Check2VarEdges.append([27, 40, 133, 153, 205, 220])
        self.__Check2VarEdges.append([28, 41, 134, 154, 206, 221])
        self.__Check2VarEdges.append([29, 42, 135, 155, 207, 222])
        self.__Check2VarEdges.append([30, 43, 136, 156, 208, 223])
        self.__Check2VarEdges.append([31, 44, 137, 157, 193, 224])
        self.__Check2VarEdges.append([21, 84, 110, 114, 179, 209, 225])
        self.__Check2VarEdges.append([22, 85, 111, 115, 180, 210, 226])
        self.__Check2VarEdges.append([23, 86, 112, 116, 181, 211, 227])
        self.__Check2VarEdges.append([24, 87, 97, 117, 182, 212, 228])
        self.__Check2VarEdges.append([25, 88, 98, 118, 183, 213, 229])
        self.__Check2VarEdges.append([26, 89, 99, 119, 184, 214, 230])
        self.__Check2VarEdges.append([27, 90, 100, 120, 185, 215, 231])
        self.__Check2VarEdges.append([28, 91, 101, 121, 186, 216, 232])
        self.__Check2VarEdges.append([29, 92, 102, 122, 187, 217, 233])
        self.__Check2VarEdges.append([30, 93, 103, 123, 188, 218, 234])
        self.__Check2VarEdges.append([31, 94, 104, 124, 189, 219, 235])
        self.__Check2VarEdges.append([32, 95, 105, 125, 190, 220, 236])
        self.__Check2VarEdges.append([17, 96, 106, 126, 191, 221, 237])
        self.__Check2VarEdges.append([18, 81, 107, 127, 192, 222, 238])
        self.__Check2VarEdges.append([19, 82, 108, 128, 177, 223, 239])
        self.__Check2VarEdges.append([20, 83, 109, 113, 178, 224, 240])
        self.__Check2VarEdges.append([53, 68, 94, 118, 177, 225, 241])
        self.__Check2VarEdges.append([54, 69, 95, 119, 178, 226, 242])
        self.__Check2VarEdges.append([55, 70, 96, 120, 179, 227, 243])
        self.__Check2VarEdges.append([56, 71, 81, 121, 180, 228, 244])
        self.__Check2VarEdges.append([57, 72, 82, 122, 181, 229, 245])
        self.__Check2VarEdges.append([58, 73, 83, 123, 182, 230, 246])
        self.__Check2VarEdges.append([59, 74, 84, 124, 183, 231, 247])
        self.__Check2VarEdges.append([60, 75, 85, 125, 184, 232, 248])
        self.__Check2VarEdges.append([61, 76, 86, 126, 185, 233, 249])
        self.__Check2VarEdges.append([62, 77, 87, 127, 186, 234, 250])
        self.__Check2VarEdges.append([63, 78, 88, 128, 187, 235, 251])
        self.__Check2VarEdges.append([64, 79, 89, 113, 188, 236, 252])
        self.__Check2VarEdges.append([49, 80, 90, 114, 189, 237, 253])
        self.__Check2VarEdges.append([50, 65, 91, 115, 190, 238, 254])
        self.__Check2VarEdges.append([51, 66, 92, 116, 191, 239, 255])
        self.__Check2VarEdges.append([52, 67, 93, 117, 192, 240, 256])
        self.__Check2VarEdges.append([11, 40, 139, 149, 241, 257])
        self.__Check2VarEdges.append([12, 41, 140, 150, 242, 258])
        self.__Check2VarEdges.append([13, 42, 141, 151, 243, 259])
        self.__Check2VarEdges.append([14, 43, 142, 152, 244, 260])
        self.__Check2VarEdges.append([15, 44, 143, 153, 245, 261])
        self.__Check2VarEdges.append([16, 45, 144, 154, 246, 262])
        self.__Check2VarEdges.append([1, 46, 129, 155, 247, 263])
        self.__Check2VarEdges.append([2, 47, 130, 156, 248, 264])
        self.__Check2VarEdges.append([3, 48, 131, 157, 249, 265])
        self.__Check2VarEdges.append([4, 33, 132, 158, 250, 266])
        self.__Check2VarEdges.append([5, 34, 133, 159, 251, 267])
        self.__Check2VarEdges.append([6, 35, 134, 160, 252, 268])
        self.__Check2VarEdges.append([7, 36, 135, 145, 253, 269])
        self.__Check2VarEdges.append([8, 37, 136, 146, 254, 270])
        self.__Check2VarEdges.append([9, 38, 137, 147, 255, 271])
        self.__Check2VarEdges.append([10, 39, 138, 148, 256, 272])
        self.__Check2VarEdges.append([39, 111, 151, 173, 257, 273])
        self.__Check2VarEdges.append([40, 112, 152, 174, 258, 274])
        self.__Check2VarEdges.append([41, 97, 153, 175, 259, 275])
        self.__Check2VarEdges.append([42, 98, 154, 176, 260, 276])
        self.__Check2VarEdges.append([43, 99, 155, 161, 261, 277])
        self.__Check2VarEdges.append([44, 100, 156, 162, 262, 278])
        self.__Check2VarEdges.append([45, 101, 157, 163, 263, 279])
        self.__Check2VarEdges.append([46, 102, 158, 164, 264, 280])
        self.__Check2VarEdges.append([47, 103, 159, 165, 265, 281])
        self.__Check2VarEdges.append([48, 104, 160, 166, 266, 282])
        self.__Check2VarEdges.append([33, 105, 145, 167, 267, 283])
        self.__Check2VarEdges.append([34, 106, 146, 168, 268, 284])
        self.__Check2VarEdges.append([35, 107, 147, 169, 269, 285])
        self.__Check2VarEdges.append([36, 108, 148, 170, 270, 286])
        self.__Check2VarEdges.append([37, 109, 149, 171, 271, 287])
        self.__Check2VarEdges.append([38, 110, 150, 172, 272, 288])
        self.__Check2VarEdges.append([72, 87, 126, 190, 193, 273, 289])
        self.__Check2VarEdges.append([73, 88, 127, 191, 194, 274, 290])
        self.__Check2VarEdges.append([74, 89, 128, 192, 195, 275, 291])
        self.__Check2VarEdges.append([75, 90, 113, 177, 196, 276, 292])
        self.__Check2VarEdges.append([76, 91, 114, 178, 197, 277, 293])
        self.__Check2VarEdges.append([77, 92, 115, 179, 198, 278, 294])
        self.__Check2VarEdges.append([78, 93, 116, 180, 199, 279, 295])
        self.__Check2VarEdges.append([79, 94, 117, 181, 200, 280, 296])
        self.__Check2VarEdges.append([80, 95, 118, 182, 201, 281, 297])
        self.__Check2VarEdges.append([65, 96, 119, 183, 202, 282, 298])
        self.__Check2VarEdges.append([66, 81, 120, 184, 203, 283, 299])
        self.__Check2VarEdges.append([67, 82, 121, 185, 204, 284, 300])
        self.__Check2VarEdges.append([68, 83, 122, 186, 205, 285, 301])
        self.__Check2VarEdges.append([69, 84, 123, 187, 206, 286, 302])
        self.__Check2VarEdges.append([70, 85, 124, 188, 207, 287, 303])
        self.__Check2VarEdges.append([71, 86, 125, 189, 208, 288, 304])
        self.__Check2VarEdges.append([48, 57, 147, 164, 289, 305])
        self.__Check2VarEdges.append([33, 58, 148, 165, 290, 306])
        self.__Check2VarEdges.append([34, 59, 149, 166, 291, 307])
        self.__Check2VarEdges.append([35, 60, 150, 167, 292, 308])
        self.__Check2VarEdges.append([36, 61, 151, 168, 293, 309])
        self.__Check2VarEdges.append([37, 62, 152, 169, 294, 310])
        self.__Check2VarEdges.append([38, 63, 153, 170, 295, 311])
        self.__Check2VarEdges.append([39, 64, 154, 171, 296, 312])
        self.__Check2VarEdges.append([40, 49, 155, 172, 297, 313])
        self.__Check2VarEdges.append([41, 50, 156, 173, 298, 314])
        self.__Check2VarEdges.append([42, 51, 157, 174, 299, 315])
        self.__Check2VarEdges.append([43, 52, 158, 175, 300, 316])
        self.__Check2VarEdges.append([44, 53, 159, 176, 301, 317])
        self.__Check2VarEdges.append([45, 54, 160, 161, 302, 318])
        self.__Check2VarEdges.append([46, 55, 145, 162, 303, 319])
        self.__Check2VarEdges.append([47, 56, 146, 163, 304, 320])
        self.__Check2VarEdges.append([18, 45, 97, 152, 305, 321])
        self.__Check2VarEdges.append([19, 46, 98, 153, 306, 322])
        self.__Check2VarEdges.append([20, 47, 99, 154, 307, 323])
        self.__Check2VarEdges.append([21, 48, 100, 155, 308, 324])
        self.__Check2VarEdges.append([22, 33, 101, 156, 309, 325])
        self.__Check2VarEdges.append([23, 34, 102, 157, 310, 326])
        self.__Check2VarEdges.append([24, 35, 103, 158, 311, 327])
        self.__Check2VarEdges.append([25, 36, 104, 159, 312, 328])
        self.__Check2VarEdges.append([26, 37, 105, 160, 313, 329])
        self.__Check2VarEdges.append([27, 38, 106, 145, 314, 330])
        self.__Check2VarEdges.append([28, 39, 107, 146, 315, 331])
        self.__Check2VarEdges.append([29, 40, 108, 147, 316, 332])
        self.__Check2VarEdges.append([30, 41, 109, 148, 317, 333])
        self.__Check2VarEdges.append([31, 42, 110, 149, 318, 334])
        self.__Check2VarEdges.append([32, 43, 111, 150, 319, 335])
        self.__Check2VarEdges.append([17, 44, 112, 151, 320, 336])
        self.__Check2VarEdges.append([3, 78, 85, 120, 185, 321, 337])
        self.__Check2VarEdges.append([4, 79, 86, 121, 186, 322, 338])
        self.__Check2VarEdges.append([5, 80, 87, 122, 187, 323, 339])
        self.__Check2VarEdges.append([6, 65, 88, 123, 188, 324, 340])
        self.__Check2VarEdges.append([7, 66, 89, 124, 189, 325, 341])
        self.__Check2VarEdges.append([8, 67, 90, 125, 190, 326, 342])
        self.__Check2VarEdges.append([9, 68, 91, 126, 191, 327, 343])
        self.__Check2VarEdges.append([10, 69, 92, 127, 192, 328, 344])
        self.__Check2VarEdges.append([11, 70, 93, 128, 177, 329, 345])
        self.__Check2VarEdges.append([12, 71, 94, 113, 178, 330, 346])
        self.__Check2VarEdges.append([13, 72, 95, 114, 179, 331, 347])
        self.__Check2VarEdges.append([14, 73, 96, 115, 180, 332, 348])
        self.__Check2VarEdges.append([15, 74, 81, 116, 181, 333, 349])
        self.__Check2VarEdges.append([16, 75, 82, 117, 182, 334, 350])
        self.__Check2VarEdges.append([1, 76, 83, 118, 183, 335, 351])
        self.__Check2VarEdges.append([2, 77, 84, 119, 184, 336, 352])
        self.__Check2VarEdges.append([96, 122, 172, 189, 337, 353])
        self.__Check2VarEdges.append([81, 123, 173, 190, 338, 354])
        self.__Check2VarEdges.append([82, 124, 174, 191, 339, 355])
        self.__Check2VarEdges.append([83, 125, 175, 192, 340, 356])
        self.__Check2VarEdges.append([84, 126, 176, 177, 341, 357])
        self.__Check2VarEdges.append([85, 127, 161, 178, 342, 358])
        self.__Check2VarEdges.append([86, 128, 162, 179, 343, 359])
        self.__Check2VarEdges.append([87, 113, 163, 180, 344, 360])
        self.__Check2VarEdges.append([88, 114, 164, 181, 345, 361])
        self.__Check2VarEdges.append([89, 115, 165, 182, 346, 362])
        self.__Check2VarEdges.append([90, 116, 166, 183, 347, 363])
        self.__Check2VarEdges.append([91, 117, 167, 184, 348, 364])
        self.__Check2VarEdges.append([92, 118, 168, 185, 349, 365])
        self.__Check2VarEdges.append([93, 119, 169, 186, 350, 366])
        self.__Check2VarEdges.append([94, 120, 170, 187, 351, 367])
        self.__Check2VarEdges.append([95, 121, 171, 188, 352, 368])
        self.__Check2VarEdges.append([34, 59, 135, 153, 353, 369])
        self.__Check2VarEdges.append([35, 60, 136, 154, 354, 370])
        self.__Check2VarEdges.append([36, 61, 137, 155, 355, 371])
        self.__Check2VarEdges.append([37, 62, 138, 156, 356, 372])
        self.__Check2VarEdges.append([38, 63, 139, 157, 357, 373])
        self.__Check2VarEdges.append([39, 64, 140, 158, 358, 374])
        self.__Check2VarEdges.append([40, 49, 141, 159, 359, 375])
        self.__Check2VarEdges.append([41, 50, 142, 160, 360, 376])
        self.__Check2VarEdges.append([42, 51, 143, 145, 361, 377])
        self.__Check2VarEdges.append([43, 52, 144, 146, 362, 378])
        self.__Check2VarEdges.append([44, 53, 129, 147, 363, 379])
        self.__Check2VarEdges.append([45, 54, 130, 148, 364, 380])
        self.__Check2VarEdges.append([46, 55, 131, 149, 365, 381])
        self.__Check2VarEdges.append([47, 56, 132, 150, 366, 382])
        self.__Check2VarEdges.append([48, 57, 133, 151, 367, 383])
        self.__Check2VarEdges.append([33, 58, 134, 152, 368, 384])
        self.__Check2VarEdges.append([8, 92, 119, 181, 194, 369])
        self.__Check2VarEdges.append([9, 93, 120, 182, 195, 370])
        self.__Check2VarEdges.append([10, 94, 121, 183, 196, 371])
        self.__Check2VarEdges.append([11, 95, 122, 184, 197, 372])
        self.__Check2VarEdges.append([12, 96, 123, 185, 198, 373])
        self.__Check2VarEdges.append([13, 81, 124, 186, 199, 374])
        self.__Check2VarEdges.append([14, 82, 125, 187, 200, 375])
        self.__Check2VarEdges.append([15, 83, 126, 188, 201, 376])
        self.__Check2VarEdges.append([16, 84, 127, 189, 202, 377])
        self.__Check2VarEdges.append([1, 85, 128, 190, 203, 378])
        self.__Check2VarEdges.append([2, 86, 113, 191, 204, 379])
        self.__Check2VarEdges.append([3, 87, 114, 192, 205, 380])
        self.__Check2VarEdges.append([4, 88, 115, 177, 206, 381])
        self.__Check2VarEdges.append([5, 89, 116, 178, 207, 382])
        self.__Check2VarEdges.append([6, 90, 117, 179, 208, 383])
        self.__Check2VarEdges.append([7, 91, 118, 180, 193, 384])
        super().__init__(self.__Check2VarEdges, None, seclength)
        self.__DepthFromRoot = 96  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class GraphTest(ccsfg.Encoding):
    def __init__(self, seclength=2):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([2, 4])
        super().__init__(self.__Check2VarEdges, [1, 2], seclength)
        self.__DepthFromRoot = 8  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


def decoder(graph, stateestimates, count):  # NEED ORDER OUTPUT IN LIKELIHOOD MAYBE
    """
    This method seeks to disambiguate codewords from node states.
    Gather local state estimates from variables nodes and retain top values in place.
    Set values of other indices within every section to zero.
    Perform belief propagation and return `count` likely codewords.
    :param graph: Bipartite graph for error correcting code.
    :param stateestimates: Local estimates from variable nodes.
    :param count: Maximum number of codewords returned.
    :return: List of likely codewords.
    """

    # Resize @var stateestimates to match local measures from variable nodes.
    stateestimates.resize(graph.getvarcount(), graph.getsparseseclength())
    thresholdedestimates = np.zeros(stateestimates.shape)

    # # NOTE: Pruning impossible paths prior to root decoding has minimal impact.
    # # CCS-AMP already encourages paths to be locally consistent.
    # # This step is extraneous and should probably be avoided.
    # topindices = []
    # hardestimates = np.zeros(stateestimates.shape)
    # idx: int
    # for idx in range(graph.getvarcount()):
    #     trailingtopindices = np.argpartition(stateestimates[idx], -64)[-64:]
    #     # Retain values corresponding to top indices and zero out other entries.
    #     # Set most likely locations to one.
    #     for topidx in trailingtopindices:
    #         hardestimates[idx, topidx] = 1 if (stateestimates[idx, topidx] != 0) else 0
    #     print(np.linalg.norm(hardestimates[idx], ord=0), end=' ')
    #     graph.setobservation(idx+1,hardestimates[idx,:])
    # print('\n')
    #
    # for iteration in range(16):    # For Graph6
    #     graph.updatechecks()  # Update Check first
    #     graph.updatevars()
    #
    # for idx in range(graph.getvarcount()):
    #     hardestimates[idx] = graph.getestimate(idx+1)
    #     print(np.linalg.norm(hardestimates[idx,:], ord=0), end=' ')

    # Retain most likely values in every section.
    idx: int
    for idx in range(graph.getvarcount()):
        # Function np.argpartition puts indices of top arguments at the end (unordered).
        # Variable @var trailingtopindices holds these arguments.
        trailingtopindices = np.argpartition(stateestimates[idx], -count)[-count:]  # CHECK count or 1024
        # Retain values corresponding to top indices and zero out other entries.
        for topidx in trailingtopindices:
            thresholdedestimates[idx, topidx] = stateestimates[idx, topidx]

    # Find `count` most likely locations in every section and zero out the rest.
    # List of candidate codewords.
    recoveredcodewords = []
    # Function np.argpartition puts indices of top arguments at the end.
    # If count differs from above argument, then call np.argpartition again because top output are not ordered.
    # Indices of `count` most likely locations in root section
    trailingtopindices = np.argpartition(thresholdedestimates[0, :], -count)[-count:]
    # Iterating through evey retained location in root section
    for topidx in trailingtopindices:
        print('Root section ID: ' + str(topidx))
        # Reset graph, including check nodes, is critical for every root location.
        graph.reset()
        rootsingleton = np.zeros(graph.getsparseseclength())
        rootsingleton[topidx] = 1 if (thresholdedestimates[0, topidx] != 0) else 0
        graph.setobservation(1, rootsingleton)
        for idx in range(1, graph.getvarcount()):
            graph.setobservation(idx + 1, thresholdedestimates[idx, :])

        ## This may only work for hierchical settings.

        # Start with full list of nodes to update.
        checknodes2update = set(graph.getchecklist())
        graph.updatechecks(checknodes2update)  # Update Check first
        varnodes2update = set(graph.getvarlist())
        graph.updatevars(varnodes2update)
        # Initialize vector of section weights
        newsectionweights0 = np.linalg.norm(graph.getestimates(), ord=0, axis=1)
        for iteration in range(graph.getmaxdepth()):  # Max depth
            sectionweights0 = np.linalg.norm(graph.getestimates(), ord=0, axis=1)
            checkneighbors = set()
            varneighbors = set()

            # Update variable nodes and check for convergence
            graph.updatevars(varnodes2update)
            for varnodeid in varnodes2update:
                currentmeasure = graph.getestimate(varnodeid)
                currentweight1 = np.linalg.norm(currentmeasure, ord=1)
                if np.isclose(currentweight1, np.amax(currentmeasure)):
                    varnodes2update = varnodes2update - {varnodeid}
                    checkneighbors.update(graph.getvarneighbors(varnodeid))
                    singleton = np.zeros(graph.getsparseseclength())
                    if np.isclose(currentweight1, 0):
                        pass
                    else:
                        singleton[np.argmax(currentmeasure)] = 1 if (thresholdedestimates[0, topidx] != 0) else 0
                    graph.setobservation(varnodeid, singleton)
            if checkneighbors != set():
                graph.updatechecks(checkneighbors)
            # print('Variable nodes to update: ' + str(varnodes2update))

            # Update check nodes and check for convergence
            graph.updatechecks(checknodes2update)
            for checknodeid in checknodes2update:
                if set(graph.getcheckneighbors(checknodeid)).isdisjoint(varnodes2update):
                    checknodes2update = checknodes2update - {checknodeid}
                    varneighbors.update(graph.getcheckneighbors(checknodeid))
            if varneighbors != set():
                graph.updatevars(varneighbors)
            # print('Check nodes to update: ' + str(checknodes2update))

            # Monitor progress and trim section, if necessary
            newsectionweights0 = np.linalg.norm(graph.getestimates(), ord=0, axis=1)
            # maxsectionlength = 1 + np.ceil(1024 * (graph.getmaxdepth() - iteration - 1)/graph.getmaxdepth()).astype(int)
            maxsectionlength = np.ceil(
                2 ** ((np.log2(128) / graph.getmaxdepth()) * (graph.getmaxdepth() - iteration - 1))).astype(int)
            if np.amin(newsectionweights0) == 0 or len(varnodes2update) == 0:
                break
            # elif np.array_equal(sectionweights0, newsectionweights0):
            elif np.amax(newsectionweights0) > maxsectionlength:
                # print('trimming')
                for varnodeid in varnodes2update:
                    currentmeasure = graph.getestimate(varnodeid)
                    currentweight0 = np.linalg.norm(currentmeasure, ord=0).astype(int)
                    if currentweight0 > maxsectionlength:
                        supportsize = maxsectionlength
                        # Function np.argpartition puts indices of top arguments at the end (unordered).
                        # Variable @var trailingtopindices holds these arguments.
                        currentobservation = graph.getobservation(varnodeid)
                        trimmedtopindices = np.argpartition(currentmeasure, -supportsize)[-supportsize:]
                        # Retain values corresponding to top indices and zero out other entries.
                        trimmedobservation = np.zeros(graph.getsparseseclength())
                        for trimmedidx in trimmedtopindices:
                            trimmedobservation[trimmedidx] = currentobservation[trimmedidx]
                            graph.setobservation(varnodeid, trimmedobservation)
                        graph.updatechecks(graph.getvarneighbors(varnodeid))
                    else:
                        pass
            # print('Weights ' + str(newsectionweights0))

        decoded = graph.getcodeword().flatten()
        decodedsum = np.sum(decoded.flatten())
        if decodedsum == graph.getvarcount():
            recoveredcodewords.append(decoded)
        elif decodedsum > graph.getvarcount():  # CHECK: Can be improved later
            print('Disambiguation failed.')
            recoveredcodewords.append(decoded)
        else:
            pass

    # Order candidates
    likelihoods = []
    for candidate in recoveredcodewords:
        isolatedvalues = np.prod((candidate, stateestimates.flatten()), axis=0)
        isolatedvalues.resize(graph.getvarcount(), graph.getsparseseclength())
        likelihoods.append(np.prod(np.amax(isolatedvalues, axis=1)))
    idxsorted = np.argsort(likelihoods)
    recoveredcodewords = [recoveredcodewords[idx] for idx in idxsorted[::-1]]
    return recoveredcodewords


def numbermatches(codewords, recoveredcodewords, maxcount=None):
    """
    Counts number of matches between `codewords` and `recoveredcodewords`.
    CHECK: Does not insure uniqueness.
    :param codewords: List of true codewords.
    :param recoveredcodewords: List of candidate codewords from most to least likely.
    :return: Number of true codewords recovered.
    """
    # Provision for scenario where candidate count is smaller than codeword count.
    if maxcount is None:
        maxcount = min(len(codewords), len(recoveredcodewords))
    else:
        maxcount = min(len(codewords), len(recoveredcodewords), maxcount)
    matchcount = 0
    for candidateindex in range(maxcount):
        candidate = recoveredcodewords[candidateindex]
        # print('Candidate codeword: ' + str(candidate))
        # print(np.equal(codewords,candidate).all(axis=1)) # Check if candidate individual codewords
        matchcount = matchcount + (np.equal(codewords, candidate).all(axis=1).any()).astype(int)  # Check if matches any
    return matchcount


def displayinfo(graph, binarysequence):
    alphabet = graph.getseclength()
    binsections = binarysequence.reshape(alphabet, -1)
    sections = []
    # for sectionindex in range(len(binarysequence)):
    # sections.append(np.sum([2**i - 1 for i in np.argmax(binsections[sectionindex])-1)
    print(sections)

#
# TestCode = Triadic8(2)
# TestCode.printgraph()
# TestCode.reset()
# infoarray = [[1, 0, 0, 0, 1, 0]]
# codeword = TestCode.encodemessage(infoarray[0])

# TestCode.printgraphcontent()
#
# NumberDevices = 1
# infoarray = np.random.randint(2, size=(NumberDevices,TestCode.getinfocount()*TestCode.getseclength()))
# infoarray = [[1, 1, 0, 0, 0, 0, 0, 0]]
# print('Information bits:\n' + str(infoarray))
# print('Signal sections:\n' + str(codewords))
# codeword = TestCode.encodemessage(infoarray[0])
# print(codeword)
# print(np.linalg.norm(codeword, ord=0))
# TestCode.printgraphcontent()

# signal = TestCode.encodesignal(infoarray)
# print('Signal reshaped:\n' + str(signal))

# testvector = np.ones((TestCode.getvarcount(),TestCode.getsparseseclength()))
# testvector[0] = np.zeros(TestCode.getsparseseclength())
# for sectionid in TestCode.getvarlist():
#     TestCode.setobservation(sectionid, testvector[sectionid-1,:])
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid).flatten() for varnodeid in TestCode.getvarlist()],axis=1))
# for iteration in range(TestCode.getmaxdepth()):    # Max depth
#     TestCode.updatechecks()
#     TestCode.updatevars()
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid) for varnodeid in TestCode.getvarlist()],axis=1))


#
# OuterCode = Graph6(4)
# OuterCode.printgraph()
#

#
# codewords = OuterCode.encodemessages(infoarray)
# print('Codewords shape: ' + str(codewords.shape))
# codeword = codewords[0]
# print('Codeword shape' + str(codeword.shape))
# notcodeword = codeword[::-1]
#
# output = OuterCode.testvalid(notcodeword.flatten())
# print('Test codeword: ' + str(np.linalg.norm(output.flatten(),ord=1)))
# print(output)
#
# # print('Estimates shape' + str(OuterCode.getestimates().shape))
# # print(np.linalg.norm(OuterCode.getestimates().flatten(),ord=1))
# # print(np.linalg.norm(output.flatten() - codeword,ord=1))
#
# originallist = codewords.copy()
# recoveredcodewords = decoder(OuterCode,signal[::-1],NumberDevices)
# print(recoveredcodewords)
#
# matches = numbermatches(originallist,recoveredcodewords)
# print(matches)
