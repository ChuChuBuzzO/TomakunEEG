# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_psi_settings.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 426)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(50, 390, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 40, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.SpinBoxRangeMin = QtWidgets.QSpinBox(Dialog)
        self.SpinBoxRangeMin.setGeometry(QtCore.QRect(40, 75, 43, 22))
        self.SpinBoxRangeMin.setObjectName("SpinBoxRangeMin")
        self.SpinBoxRangeMax = QtWidgets.QSpinBox(Dialog)
        self.SpinBoxRangeMax.setGeometry(QtCore.QRect(160, 75, 43, 22))
        self.SpinBoxRangeMax.setObjectName("SpinBoxRangeMax")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(110, 80, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 80, 62, 15))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 120, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(20, 180, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.PushButtonChangeSeed = QtWidgets.QPushButton(Dialog)
        self.PushButtonChangeSeed.setGeometry(QtCore.QRect(290, 150, 93, 28))
        self.PushButtonChangeSeed.setObjectName("PushButtonChangeSeed")
        self.LabelSeed = QtWidgets.QLabel(Dialog)
        self.LabelSeed.setGeometry(QtCore.QRect(30, 150, 241, 16))
        self.LabelSeed.setText("")
        self.LabelSeed.setObjectName("LabelSeed")
        self.LabelTarget = QtWidgets.QLabel(Dialog)
        self.LabelTarget.setGeometry(QtCore.QRect(30, 210, 241, 16))
        self.LabelTarget.setText("")
        self.LabelTarget.setObjectName("LabelTarget")
        self.PushButtonChangeTarget = QtWidgets.QPushButton(Dialog)
        self.PushButtonChangeTarget.setGeometry(QtCore.QRect(290, 210, 93, 28))
        self.PushButtonChangeTarget.setObjectName("PushButtonChangeTarget")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(20, 270, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(20, 330, 321, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(20, 360, 161, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(270, 40, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox.setGeometry(QtCore.QRect(290, 75, 62, 22))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(360, 80, 62, 15))
        self.label_10.setObjectName("label_10")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Frequency range"))
        self.label_2.setText(_translate("Dialog", "～"))
        self.label_3.setText(_translate("Dialog", "Hz"))
        self.label_4.setText(_translate("Dialog", "Seed"))
        self.label_5.setText(_translate("Dialog", "Target"))
        self.PushButtonChangeSeed.setText(_translate("Dialog", "Change"))
        self.PushButtonChangeTarget.setText(_translate("Dialog", "Change"))
        self.label_6.setText(_translate("Dialog", "Margin: 1s- 1s"))
        self.label_7.setText(_translate("Dialog", "Key \"a\": PSI Selecting Mode"))
        self.label_8.setText(_translate("Dialog", "Key \"c\": Calculate PSI"))
        self.label_9.setText(_translate("Dialog", "Step"))
        self.label_10.setText(_translate("Dialog", "Hz"))
