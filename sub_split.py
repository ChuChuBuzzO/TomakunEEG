# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_split.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(935, 631)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setEnabled(False)
        self.buttonBox.setGeometry(QtCore.QRect(560, 580, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(420, 20, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(460, 70, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.CheckButton = QtWidgets.QPushButton(Dialog)
        self.CheckButton.setGeometry(QtCore.QRect(40, 580, 93, 28))
        self.CheckButton.setObjectName("CheckButton")
        self.ConvButton = QtWidgets.QPushButton(Dialog)
        self.ConvButton.setGeometry(QtCore.QRect(140, 580, 93, 28))
        self.ConvButton.setObjectName("ConvButton")
        self.TextBrowserA = QtWidgets.QTextBrowser(Dialog)
        self.TextBrowserA.setGeometry(QtCore.QRect(60, 360, 361, 181))
        self.TextBrowserA.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard|QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextBrowserInteraction|QtCore.Qt.TextEditable|QtCore.Qt.TextEditorInteraction|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.TextBrowserA.setObjectName("TextBrowserA")
        self.TextBrowserB = QtWidgets.QTextBrowser(Dialog)
        self.TextBrowserB.setGeometry(QtCore.QRect(510, 360, 341, 181))
        self.TextBrowserB.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard|QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextBrowserInteraction|QtCore.Qt.TextEditable|QtCore.Qt.TextEditorInteraction|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.TextBrowserB.setObjectName("TextBrowserB")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(450, 420, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.WarningLabel = QtWidgets.QLabel(Dialog)
        self.WarningLabel.setGeometry(QtCore.QRect(260, 590, 381, 16))
        self.WarningLabel.setText("")
        self.WarningLabel.setObjectName("WarningLabel")
        self.TextBrowserBefore = QtWidgets.QTextBrowser(Dialog)
        self.TextBrowserBefore.setGeometry(QtCore.QRect(60, 100, 791, 192))
        self.TextBrowserBefore.setObjectName("TextBrowserBefore")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(450, 290, 62, 111))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Split data"))
        self.label_2.setText(_translate("Dialog", "OK?"))
        self.CheckButton.setText(_translate("Dialog", "Check"))
        self.ConvButton.setText(_translate("Dialog", "Easy Paste"))
        self.label_3.setText(_translate("Dialog", "⇔"))
        self.label_4.setText(_translate("Dialog", "↓"))
