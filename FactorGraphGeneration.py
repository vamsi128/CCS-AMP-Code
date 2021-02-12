__author__ = 'JF Chamberland'

import ccsfg as ccsfg
import numpy as np

class WIFI_648_540(ccsfg.Encoding):
    def __init__(self, seclength=1):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([18, 41, 63, 103, 118, 139, 181, 202, 227, 244, 275, 313, 344, 354, 384, 416, 459, 479, 500, 527, 542, 568])
        self.__Check2VarEdges.append([19, 42, 64, 104, 119, 140, 182, 203, 228, 245, 276, 314, 345, 355, 385, 417, 433, 480, 501, 528, 543, 569])
        self.__Check2VarEdges.append([20, 43, 65, 105, 120, 141, 183, 204, 229, 246, 277, 315, 346, 356, 386, 418, 434, 481, 502, 529, 544, 570])
        self.__Check2VarEdges.append([21, 44, 66, 106, 121, 142, 184, 205, 230, 247, 278, 316, 347, 357, 387, 419, 435, 482, 503, 530, 545, 571])
        self.__Check2VarEdges.append([22, 45, 67, 107, 122, 143, 185, 206, 231, 248, 279, 317, 348, 358, 388, 420, 436, 483, 504, 531, 546, 572])
        self.__Check2VarEdges.append([23, 46, 68, 108, 123, 144, 186, 207, 232, 249, 280, 318, 349, 359, 389, 421, 437, 484, 505, 532, 547, 573])
        self.__Check2VarEdges.append([24, 47, 69, 82, 124, 145, 187, 208, 233, 250, 281, 319, 350, 360, 390, 422, 438, 485, 506, 533, 548, 574])
        self.__Check2VarEdges.append([25, 48, 70, 83, 125, 146, 188, 209, 234, 251, 282, 320, 351, 361, 391, 423, 439, 486, 507, 534, 549, 575])
        self.__Check2VarEdges.append([26, 49, 71, 84, 126, 147, 189, 210, 235, 252, 283, 321, 325, 362, 392, 424, 440, 460, 508, 535, 550, 576])
        self.__Check2VarEdges.append([27, 50, 72, 85, 127, 148, 163, 211, 236, 253, 284, 322, 326, 363, 393, 425, 441, 461, 509, 536, 551, 577])
        self.__Check2VarEdges.append([1, 51, 73, 86, 128, 149, 164, 212, 237, 254, 285, 323, 327, 364, 394, 426, 442, 462, 510, 537, 552, 578])
        self.__Check2VarEdges.append([2, 52, 74, 87, 129, 150, 165, 213, 238, 255, 286, 324, 328, 365, 395, 427, 443, 463, 511, 538, 553, 579])
        self.__Check2VarEdges.append([3, 53, 75, 88, 130, 151, 166, 214, 239, 256, 287, 298, 329, 366, 396, 428, 444, 464, 512, 539, 554, 580])
        self.__Check2VarEdges.append([4, 54, 76, 89, 131, 152, 167, 215, 240, 257, 288, 299, 330, 367, 397, 429, 445, 465, 513, 540, 555, 581])
        self.__Check2VarEdges.append([5, 28, 77, 90, 132, 153, 168, 216, 241, 258, 289, 300, 331, 368, 398, 430, 446, 466, 487, 514, 556, 582])
        self.__Check2VarEdges.append([6, 29, 78, 91, 133, 154, 169, 190, 242, 259, 290, 301, 332, 369, 399, 431, 447, 467, 488, 515, 557, 583])
        self.__Check2VarEdges.append([7, 30, 79, 92, 134, 155, 170, 191, 243, 260, 291, 302, 333, 370, 400, 432, 448, 468, 489, 516, 558, 584])
        self.__Check2VarEdges.append([8, 31, 80, 93, 135, 156, 171, 192, 217, 261, 292, 303, 334, 371, 401, 406, 449, 469, 490, 517, 559, 585])
        self.__Check2VarEdges.append([9, 32, 81, 94, 109, 157, 172, 193, 218, 262, 293, 304, 335, 372, 402, 407, 450, 470, 491, 518, 560, 586])
        self.__Check2VarEdges.append([10, 33, 55, 95, 110, 158, 173, 194, 219, 263, 294, 305, 336, 373, 403, 408, 451, 471, 492, 519, 561, 587])
        self.__Check2VarEdges.append([11, 34, 56, 96, 111, 159, 174, 195, 220, 264, 295, 306, 337, 374, 404, 409, 452, 472, 493, 520, 562, 588])
        self.__Check2VarEdges.append([12, 35, 57, 97, 112, 160, 175, 196, 221, 265, 296, 307, 338, 375, 405, 410, 453, 473, 494, 521, 563, 589])
        self.__Check2VarEdges.append([13, 36, 58, 98, 113, 161, 176, 197, 222, 266, 297, 308, 339, 376, 379, 411, 454, 474, 495, 522, 564, 590])
        self.__Check2VarEdges.append([14, 37, 59, 99, 114, 162, 177, 198, 223, 267, 271, 309, 340, 377, 380, 412, 455, 475, 496, 523, 565, 591])
        self.__Check2VarEdges.append([15, 38, 60, 100, 115, 136, 178, 199, 224, 268, 272, 310, 341, 378, 381, 413, 456, 476, 497, 524, 566, 592])
        self.__Check2VarEdges.append([16, 39, 61, 101, 116, 137, 179, 200, 225, 269, 273, 311, 342, 352, 382, 414, 457, 477, 498, 525, 567, 593])
        self.__Check2VarEdges.append([17, 40, 62, 102, 117, 138, 180, 201, 226, 270, 274, 312, 343, 353, 383, 415, 458, 478, 499, 526, 541, 594])
        self.__Check2VarEdges.append([4, 40, 66, 96, 120, 161, 168, 208, 217, 253, 273, 324, 351, 362, 403, 413, 447, 480, 491, 516, 568, 595])
        self.__Check2VarEdges.append([5, 41, 67, 97, 121, 162, 169, 209, 218, 254, 274, 298, 325, 363, 404, 414, 448, 481, 492, 517, 569, 596])
        self.__Check2VarEdges.append([6, 42, 68, 98, 122, 136, 170, 210, 219, 255, 275, 299, 326, 364, 405, 415, 449, 482, 493, 518, 570, 597])
        self.__Check2VarEdges.append([7, 43, 69, 99, 123, 137, 171, 211, 220, 256, 276, 300, 327, 365, 379, 416, 450, 483, 494, 519, 571, 598])
        self.__Check2VarEdges.append([8, 44, 70, 100, 124, 138, 172, 212, 221, 257, 277, 301, 328, 366, 380, 417, 451, 484, 495, 520, 572, 599])
        self.__Check2VarEdges.append([9, 45, 71, 101, 125, 139, 173, 213, 222, 258, 278, 302, 329, 367, 381, 418, 452, 485, 496, 521, 573, 600])
        self.__Check2VarEdges.append([10, 46, 72, 102, 126, 140, 174, 214, 223, 259, 279, 303, 330, 368, 382, 419, 453, 486, 497, 522, 574, 601])
        self.__Check2VarEdges.append([11, 47, 73, 103, 127, 141, 175, 215, 224, 260, 280, 304, 331, 369, 383, 420, 454, 460, 498, 523, 575, 602])
        self.__Check2VarEdges.append([12, 48, 74, 104, 128, 142, 176, 216, 225, 261, 281, 305, 332, 370, 384, 421, 455, 461, 499, 524, 576, 603])
        self.__Check2VarEdges.append([13, 49, 75, 105, 129, 143, 177, 190, 226, 262, 282, 306, 333, 371, 385, 422, 456, 462, 500, 525, 577, 604])
        self.__Check2VarEdges.append([14, 50, 76, 106, 130, 144, 178, 191, 227, 263, 283, 307, 334, 372, 386, 423, 457, 463, 501, 526, 578, 605])
        self.__Check2VarEdges.append([15, 51, 77, 107, 131, 145, 179, 192, 228, 264, 284, 308, 335, 373, 387, 424, 458, 464, 502, 527, 579, 606])
        self.__Check2VarEdges.append([16, 52, 78, 108, 132, 146, 180, 193, 229, 265, 285, 309, 336, 374, 388, 425, 459, 465, 503, 528, 580, 607])
        self.__Check2VarEdges.append([17, 53, 79, 82, 133, 147, 181, 194, 230, 266, 286, 310, 337, 375, 389, 426, 433, 466, 504, 529, 581, 608])
        self.__Check2VarEdges.append([18, 54, 80, 83, 134, 148, 182, 195, 231, 267, 287, 311, 338, 376, 390, 427, 434, 467, 505, 530, 582, 609])
        self.__Check2VarEdges.append([19, 28, 81, 84, 135, 149, 183, 196, 232, 268, 288, 312, 339, 377, 391, 428, 435, 468, 506, 531, 583, 610])
        self.__Check2VarEdges.append([20, 29, 55, 85, 109, 150, 184, 197, 233, 269, 289, 313, 340, 378, 392, 429, 436, 469, 507, 532, 584, 611])
        self.__Check2VarEdges.append([21, 30, 56, 86, 110, 151, 185, 198, 234, 270, 290, 314, 341, 352, 393, 430, 437, 470, 508, 533, 585, 612])
        self.__Check2VarEdges.append([22, 31, 57, 87, 111, 152, 186, 199, 235, 244, 291, 315, 342, 353, 394, 431, 438, 471, 509, 534, 586, 613])
        self.__Check2VarEdges.append([23, 32, 58, 88, 112, 153, 187, 200, 236, 245, 292, 316, 343, 354, 395, 432, 439, 472, 510, 535, 587, 614])
        self.__Check2VarEdges.append([24, 33, 59, 89, 113, 154, 188, 201, 237, 246, 293, 317, 344, 355, 396, 406, 440, 473, 511, 536, 588, 615])
        self.__Check2VarEdges.append([25, 34, 60, 90, 114, 155, 189, 202, 238, 247, 294, 318, 345, 356, 397, 407, 441, 474, 512, 537, 589, 616])
        self.__Check2VarEdges.append([26, 35, 61, 91, 115, 156, 163, 203, 239, 248, 295, 319, 346, 357, 398, 408, 442, 475, 513, 538, 590, 617])
        self.__Check2VarEdges.append([27, 36, 62, 92, 116, 157, 164, 204, 240, 249, 296, 320, 347, 358, 399, 409, 443, 476, 487, 539, 591, 618])
        self.__Check2VarEdges.append([1, 37, 63, 93, 117, 158, 165, 205, 241, 250, 297, 321, 348, 359, 400, 410, 444, 477, 488, 540, 592, 619])
        self.__Check2VarEdges.append([2, 38, 64, 94, 118, 159, 166, 206, 242, 251, 271, 322, 349, 360, 401, 411, 445, 478, 489, 514, 593, 620])
        self.__Check2VarEdges.append([3, 39, 65, 95, 119, 160, 167, 207, 243, 252, 272, 323, 350, 361, 402, 412, 446, 479, 490, 515, 594, 621])
        self.__Check2VarEdges.append([23, 44, 59, 85, 119, 157, 175, 195, 238, 258, 290, 303, 360, 384, 424, 444, 465, 492, 529, 541, 595, 622])
        self.__Check2VarEdges.append([24, 45, 60, 86, 120, 158, 176, 196, 239, 259, 291, 304, 361, 385, 425, 445, 466, 493, 530, 542, 596, 623])
        self.__Check2VarEdges.append([25, 46, 61, 87, 121, 159, 177, 197, 240, 260, 292, 305, 362, 386, 426, 446, 467, 494, 531, 543, 597, 624])
        self.__Check2VarEdges.append([26, 47, 62, 88, 122, 160, 178, 198, 241, 261, 293, 306, 363, 387, 427, 447, 468, 495, 532, 544, 598, 625])
        self.__Check2VarEdges.append([27, 48, 63, 89, 123, 161, 179, 199, 242, 262, 294, 307, 364, 388, 428, 448, 469, 496, 533, 545, 599, 626])
        self.__Check2VarEdges.append([1, 49, 64, 90, 124, 162, 180, 200, 243, 263, 295, 308, 365, 389, 429, 449, 470, 497, 534, 546, 600, 627])
        self.__Check2VarEdges.append([2, 50, 65, 91, 125, 136, 181, 201, 217, 264, 296, 309, 366, 390, 430, 450, 471, 498, 535, 547, 601, 628])
        self.__Check2VarEdges.append([3, 51, 66, 92, 126, 137, 182, 202, 218, 265, 297, 310, 367, 391, 431, 451, 472, 499, 536, 548, 602, 629])
        self.__Check2VarEdges.append([4, 52, 67, 93, 127, 138, 183, 203, 219, 266, 271, 311, 368, 392, 432, 452, 473, 500, 537, 549, 603, 630])
        self.__Check2VarEdges.append([5, 53, 68, 94, 128, 139, 184, 204, 220, 267, 272, 312, 369, 393, 406, 453, 474, 501, 538, 550, 604, 631])
        self.__Check2VarEdges.append([6, 54, 69, 95, 129, 140, 185, 205, 221, 268, 273, 313, 370, 394, 407, 454, 475, 502, 539, 551, 605, 632])
        self.__Check2VarEdges.append([7, 28, 70, 96, 130, 141, 186, 206, 222, 269, 274, 314, 371, 395, 408, 455, 476, 503, 540, 552, 606, 633])
        self.__Check2VarEdges.append([8, 29, 71, 97, 131, 142, 187, 207, 223, 270, 275, 315, 372, 396, 409, 456, 477, 504, 514, 553, 607, 634])
        self.__Check2VarEdges.append([9, 30, 72, 98, 132, 143, 188, 208, 224, 244, 276, 316, 373, 397, 410, 457, 478, 505, 515, 554, 608, 635])
        self.__Check2VarEdges.append([10, 31, 73, 99, 133, 144, 189, 209, 225, 245, 277, 317, 374, 398, 411, 458, 479, 506, 516, 555, 609, 636])
        self.__Check2VarEdges.append([11, 32, 74, 100, 134, 145, 163, 210, 226, 246, 278, 318, 375, 399, 412, 459, 480, 507, 517, 556, 610, 637])
        self.__Check2VarEdges.append([12, 33, 75, 101, 135, 146, 164, 211, 227, 247, 279, 319, 376, 400, 413, 433, 481, 508, 518, 557, 611, 638])
        self.__Check2VarEdges.append([13, 34, 76, 102, 109, 147, 165, 212, 228, 248, 280, 320, 377, 401, 414, 434, 482, 509, 519, 558, 612, 639])
        self.__Check2VarEdges.append([14, 35, 77, 103, 110, 148, 166, 213, 229, 249, 281, 321, 378, 402, 415, 435, 483, 510, 520, 559, 613, 640])
        self.__Check2VarEdges.append([15, 36, 78, 104, 111, 149, 167, 214, 230, 250, 282, 322, 352, 403, 416, 436, 484, 511, 521, 560, 614, 641])
        self.__Check2VarEdges.append([16, 37, 79, 105, 112, 150, 168, 215, 231, 251, 283, 323, 353, 404, 417, 437, 485, 512, 522, 561, 615, 642])
        self.__Check2VarEdges.append([17, 38, 80, 106, 113, 151, 169, 216, 232, 252, 284, 324, 354, 405, 418, 438, 486, 513, 523, 562, 616, 643])
        self.__Check2VarEdges.append([18, 39, 81, 107, 114, 152, 170, 190, 233, 253, 285, 298, 355, 379, 419, 439, 460, 487, 524, 563, 617, 644])
        self.__Check2VarEdges.append([19, 40, 55, 108, 115, 153, 171, 191, 234, 254, 286, 299, 356, 380, 420, 440, 461, 488, 525, 564, 618, 645])
        self.__Check2VarEdges.append([20, 41, 56, 82, 116, 154, 172, 192, 235, 255, 287, 300, 357, 381, 421, 441, 462, 489, 526, 565, 619, 646])
        self.__Check2VarEdges.append([21, 42, 57, 83, 117, 155, 173, 193, 236, 256, 288, 301, 358, 382, 422, 442, 463, 490, 527, 566, 620, 647])
        self.__Check2VarEdges.append([22, 43, 58, 84, 118, 156, 174, 194, 237, 257, 289, 302, 359, 383, 423, 443, 464, 491, 528, 567, 621, 648])
        self.__Check2VarEdges.append([8, 35, 69, 96, 113, 152, 179, 214, 241, 254, 272, 305, 340, 358, 389, 432, 441, 478, 508, 528, 542, 622])
        self.__Check2VarEdges.append([9, 36, 70, 97, 114, 153, 180, 215, 242, 255, 273, 306, 341, 359, 390, 406, 442, 479, 509, 529, 543, 623])
        self.__Check2VarEdges.append([10, 37, 71, 98, 115, 154, 181, 216, 243, 256, 274, 307, 342, 360, 391, 407, 443, 480, 510, 530, 544, 624])
        self.__Check2VarEdges.append([11, 38, 72, 99, 116, 155, 182, 190, 217, 257, 275, 308, 343, 361, 392, 408, 444, 481, 511, 531, 545, 625])
        self.__Check2VarEdges.append([12, 39, 73, 100, 117, 156, 183, 191, 218, 258, 276, 309, 344, 362, 393, 409, 445, 482, 512, 532, 546, 626])
        self.__Check2VarEdges.append([13, 40, 74, 101, 118, 157, 184, 192, 219, 259, 277, 310, 345, 363, 394, 410, 446, 483, 513, 533, 547, 627])
        self.__Check2VarEdges.append([14, 41, 75, 102, 119, 158, 185, 193, 220, 260, 278, 311, 346, 364, 395, 411, 447, 484, 487, 534, 548, 628])
        self.__Check2VarEdges.append([15, 42, 76, 103, 120, 159, 186, 194, 221, 261, 279, 312, 347, 365, 396, 412, 448, 485, 488, 535, 549, 629])
        self.__Check2VarEdges.append([16, 43, 77, 104, 121, 160, 187, 195, 222, 262, 280, 313, 348, 366, 397, 413, 449, 486, 489, 536, 550, 630])
        self.__Check2VarEdges.append([17, 44, 78, 105, 122, 161, 188, 196, 223, 263, 281, 314, 349, 367, 398, 414, 450, 460, 490, 537, 551, 631])
        self.__Check2VarEdges.append([18, 45, 79, 106, 123, 162, 189, 197, 224, 264, 282, 315, 350, 368, 399, 415, 451, 461, 491, 538, 552, 632])
        self.__Check2VarEdges.append([19, 46, 80, 107, 124, 136, 163, 198, 225, 265, 283, 316, 351, 369, 400, 416, 452, 462, 492, 539, 553, 633])
        self.__Check2VarEdges.append([20, 47, 81, 108, 125, 137, 164, 199, 226, 266, 284, 317, 325, 370, 401, 417, 453, 463, 493, 540, 554, 634])
        self.__Check2VarEdges.append([21, 48, 55, 82, 126, 138, 165, 200, 227, 267, 285, 318, 326, 371, 402, 418, 454, 464, 494, 514, 555, 635])
        self.__Check2VarEdges.append([22, 49, 56, 83, 127, 139, 166, 201, 228, 268, 286, 319, 327, 372, 403, 419, 455, 465, 495, 515, 556, 636])
        self.__Check2VarEdges.append([23, 50, 57, 84, 128, 140, 167, 202, 229, 269, 287, 320, 328, 373, 404, 420, 456, 466, 496, 516, 557, 637])
        self.__Check2VarEdges.append([24, 51, 58, 85, 129, 141, 168, 203, 230, 270, 288, 321, 329, 374, 405, 421, 457, 467, 497, 517, 558, 638])
        self.__Check2VarEdges.append([25, 52, 59, 86, 130, 142, 169, 204, 231, 244, 289, 322, 330, 375, 379, 422, 458, 468, 498, 518, 559, 639])
        self.__Check2VarEdges.append([26, 53, 60, 87, 131, 143, 170, 205, 232, 245, 290, 323, 331, 376, 380, 423, 459, 469, 499, 519, 560, 640])
        self.__Check2VarEdges.append([27, 54, 61, 88, 132, 144, 171, 206, 233, 246, 291, 324, 332, 377, 381, 424, 433, 470, 500, 520, 561, 641])
        self.__Check2VarEdges.append([1, 28, 62, 89, 133, 145, 172, 207, 234, 247, 292, 298, 333, 378, 382, 425, 434, 471, 501, 521, 562, 642])
        self.__Check2VarEdges.append([2, 29, 63, 90, 134, 146, 173, 208, 235, 248, 293, 299, 334, 352, 383, 426, 435, 472, 502, 522, 563, 643])
        self.__Check2VarEdges.append([3, 30, 64, 91, 135, 147, 174, 209, 236, 249, 294, 300, 335, 353, 384, 427, 436, 473, 503, 523, 564, 644])
        self.__Check2VarEdges.append([4, 31, 65, 92, 109, 148, 175, 210, 237, 250, 295, 301, 336, 354, 385, 428, 437, 474, 504, 524, 565, 645])
        self.__Check2VarEdges.append([5, 32, 66, 93, 110, 149, 176, 211, 238, 251, 296, 302, 337, 355, 386, 429, 438, 475, 505, 525, 566, 646])
        self.__Check2VarEdges.append([6, 33, 67, 94, 111, 150, 177, 212, 239, 252, 297, 303, 338, 356, 387, 430, 439, 476, 506, 526, 567, 647])
        self.__Check2VarEdges.append([7, 34, 68, 95, 112, 151, 178, 213, 240, 253, 271, 304, 339, 357, 388, 431, 440, 477, 507, 527, 541, 648])
        self.__SystematicVars = list(range(1,540+1))
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DMIN = 8

    def getmaxdepth(self):
        return self.__DMIN

class Graph64(ccsfg.Encoding):
    def __init__(self, seclength=4):
        # https://www.uni-kl.de/fileadmin/chaco/public/alists_nonbinary/LDPC_N512_K256_GF256_d2_exp.alist
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([13, 25, 37, 61])
        self.__Check2VarEdges.append([14, 26, 38, 62])
        self.__Check2VarEdges.append([15, 27, 39, 63])
        self.__Check2VarEdges.append([16, 28, 40, 64])
        self.__Check2VarEdges.append([13, 17, 33, 57])
        self.__Check2VarEdges.append([14, 18, 34, 58])
        self.__Check2VarEdges.append([15, 19, 35, 59])
        self.__Check2VarEdges.append([16, 20, 36, 60])
        self.__Check2VarEdges.append([5, 21, 27, 53])
        self.__Check2VarEdges.append([6, 22, 28, 54])
        self.__Check2VarEdges.append([7, 23, 25, 55])
        self.__Check2VarEdges.append([8, 24, 26, 56])
        self.__Check2VarEdges.append([1, 29, 41, 61])
        self.__Check2VarEdges.append([2, 30, 42, 62])
        self.__Check2VarEdges.append([3, 31, 43, 63])
        self.__Check2VarEdges.append([4, 32, 44, 64])
        self.__Check2VarEdges.append([9, 35, 44, 55])
        self.__Check2VarEdges.append([10, 36, 41, 56])
        self.__Check2VarEdges.append([11, 33, 42, 53])
        self.__Check2VarEdges.append([12, 34, 43, 54])
        self.__Check2VarEdges.append([1, 24, 45, 59])
        self.__Check2VarEdges.append([2, 21, 46, 60])
        self.__Check2VarEdges.append([3, 22, 47, 57])
        self.__Check2VarEdges.append([4, 23, 48, 58])
        self.__Check2VarEdges.append([5, 18, 32, 49])
        self.__Check2VarEdges.append([6, 19, 29, 50])
        self.__Check2VarEdges.append([7, 20, 30, 51])
        self.__Check2VarEdges.append([8, 17, 31, 52])
        self.__Check2VarEdges.append([9, 37, 46, 52])
        self.__Check2VarEdges.append([10, 38, 47, 49])
        self.__Check2VarEdges.append([11, 39, 48, 50])
        self.__Check2VarEdges.append([12, 40, 45, 51])

        self.__SystematicVars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                 25, 26, 27, 28, 29, 30, 31, 32]
        super().__init__(self.__Check2VarEdges, self.__SystematicVars, seclength)
        self.__DepthFromRoot = 16  # CHECK

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


class Graph8(ccsfg.Encoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([4, 5, 6])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([10, 11, 12])
        self.__Check2VarEdges.append([1, 7, 13])
        self.__Check2VarEdges.append([2, 10, 14])
        self.__Check2VarEdges.append([4, 8, 15])
        self.__Check2VarEdges.append([5, 11, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 4, 5, 7, 8, 10, 11], seclength)
        self.__DepthFromRoot = 32  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph6(ccsfg.Encoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 7])
        self.__Check2VarEdges.append([1, 3, 8])
        self.__Check2VarEdges.append([7, 4, 9])
        self.__Check2VarEdges.append([1, 12])
        self.__Check2VarEdges.append([2, 5, 11])
        self.__Check2VarEdges.append([2, 6, 10])
        self.__Check2VarEdges.append([3, 13])
        self.__Check2VarEdges.append([4, 14])
        self.__Check2VarEdges.append([5, 15])
        self.__Check2VarEdges.append([6, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 3, 4, 5, 6], seclength)
        self.__DepthFromRoot = 32  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph62(ccsfg.Encoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([4, 5, 6])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([1, 7, 10])
        self.__Check2VarEdges.append([2, 5, 11])
        self.__Check2VarEdges.append([4, 8, 12])
        self.__Check2VarEdges.append([3, 12, 13])
        self.__Check2VarEdges.append([6, 10, 14])
        self.__Check2VarEdges.append([9, 11, 15])
        self.__Check2VarEdges.append([13, 14, 15, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 4, 5, 7, 8], seclength)
        self.__DepthFromRoot = 20  # CHECK

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
# TestCode = WIFI_648_540(1)
# TestCode.printgraph()
# TestCode.reset()
# infoarray = [[1, 0, 0, 0, 1, 0]]
# codeword = TestCode.encodemessage(infoarray[0])

# TestCode.printgraphcontent()
#
# NumberDevices = 1
# infoarray = np.random.randint(2, size=(NumberDevices,TestCode.getinfocount()*TestCode.getseclength()))
# infoarray = [[1, 0, 0, 0, 1, 0]]
# print('Information bits:\n' + str(infoarray))
# print('Signal sections:\n' + str(codewords))
# codeword = TestCode.encodemessage(infoarray[0])
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
