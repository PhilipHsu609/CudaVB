Imports System.Drawing
Imports System.Drawing.Imaging
Imports System.Runtime.InteropServices

Module Program
    Private Const dllFile As String = "../../../../x64/Release/CudaDLL.dll"

    ' 宣告 DLL 函式
    <DllImport(dllFile, EntryPoint:="deviceCount")>
    Public Function deviceCount() As Integer
    End Function

    <DllImport(dllFile, EntryPoint:="cudaAlloc")>
    Public Function cudaMalloc(ByRef ptr As IntPtr, ByVal size As Long) As IntPtr
    End Function

    <DllImport(dllFile, EntryPoint:="cudaRelease")>
    Public Function cudaFree(ByVal ptr As IntPtr) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="toGPU")>
    Public Function toGPU(ByVal cpuPtr As Byte(), ByVal gpuPtr As IntPtr, ByVal size As Long) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="toGPU")>
    Public Function toGPUi(ByVal cpuPtr As Integer(), ByVal gpuPtr As IntPtr, ByVal size As Long) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="toCPU")>
    Public Function toCPU(ByVal gpuPtr As IntPtr, ByVal cpuPtr As Byte(), ByVal size As Long) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="toCPU")>
    Public Function toCPUi(ByVal gpuPtr As IntPtr, ByVal cpuPtr As Integer(), ByVal size As Long) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="conv2DCuda")>
    Public Sub convolution(ByVal src As IntPtr, ByVal dst As IntPtr, ByVal channels As Integer, ByVal width As Integer, ByVal height As Integer, ByVal kernel As Single(), ByVal kernelSize As Integer)
    End Sub

    <DllImport(dllFile, EntryPoint:="binarizeCuda")>
    Public Sub binarize(ByVal src As IntPtr, ByVal dst As IntPtr, ByVal width As Integer, ByVal height As Integer, ByVal threshold As Byte)
    End Sub

    <DllImport(dllFile, EntryPoint:="connectedComponentsCuda")>
    Public Sub connectedComponents(ByVal src As IntPtr, ByVal dst As IntPtr, ByVal width As Integer, ByVal height As Integer)
    End Sub

    Sub test(Arg As String())
        Dim bytes(19) As Byte
        BitConverter.GetBytes(5).CopyTo(bytes, 0)
        BitConverter.GetBytes(4).CopyTo(bytes, 4)
        BitConverter.GetBytes(3).CopyTo(bytes, 8)
        BitConverter.GetBytes(2).CopyTo(bytes, 12)
        BitConverter.GetBytes(1).CopyTo(bytes, 16)

        Dim devBytes As IntPtr = Nothing

        cudaMalloc(devBytes, 5 * 4)

        toGPU(bytes, devBytes, 5 * 4)

        Dim ints(4) As Integer

        toCPUi(devBytes, ints, 5 * 4)

        For i As Integer = 0 To 4
            Console.WriteLine(ints(i))
        Next

    End Sub

    Sub Main(Arg As String())
        Console.WriteLine("CUDA device count: " & deviceCount())

        Dim img As Bitmap = OpenImage("../../../image/lena_gray.bmp")
        Dim channels As Integer = 1

        ' bitmap to byte array
        Dim bmpData As Imaging.BitmapData = img.LockBits(New Rectangle(0, 0, img.Width, img.Height), Imaging.ImageLockMode.ReadWrite, img.PixelFormat)
        Dim ptr As IntPtr = bmpData.Scan0
        Dim bytes As Integer = Math.Abs(bmpData.Stride) * img.Height
        Dim src(bytes - 1) As Byte
        Marshal.Copy(ptr, src, 0, bytes)
        img.UnlockBits(bmpData)

        ' 宣告指向 GPU 記憶體的指標
        Dim devSrc As IntPtr = Nothing
        Dim devBW As IntPtr = Nothing
        Dim devLabel As IntPtr = Nothing

        ' 分配 GPU 記憶體
        cudaMalloc(devSrc, bytes)
        cudaMalloc(devBW, bytes)
        cudaMalloc(devLabel, bytes * 4)

        ' byte array 複製到 GPU 記憶體中
        toGPU(src, devSrc, bytes)

        binarize(devSrc, devBW, img.Width, img.Height, 128)
        connectedComponents(devBW, devLabel, img.Width, img.Height)

        ' 從 GPU 記憶體中複製資料回來
        Dim dst(bytes - 1) As Integer
        toCPUi(devLabel, dst, bytes * 4)

        For i As Integer = 0 To bytes - 1
            If dst(i) <> 0 Then
                Console.WriteLine(dst(i))
            End If
        Next

        ' 複製 byte array 到新的 bitmap
        'Dim img2 As New Bitmap(img.Width, img.Height, img.PixelFormat)
        'Dim bmpData2 As Imaging.BitmapData = img2.LockBits(New Rectangle(0, 0, img.Width, img.Height), Imaging.ImageLockMode.ReadWrite, img2.PixelFormat)
        'Dim ptr2 As IntPtr = bmpData2.Scan0
        'Marshal.Copy(dst, 0, ptr2, bytes)
        'img2.UnlockBits(bmpData2)

        'WriteImage("../../../image/lena_new.bmp", img2)

        ' 釋放 GPU 記憶體
        cudaFree(devSrc)
        cudaFree(devBW)
        cudaFree(devLabel)
    End Sub

    Function OpenImage(filename As String) As Bitmap
        Dim byteArr() As Byte = FileIO.FileSystem.ReadAllBytes(filename)
        Dim ms As New IO.MemoryStream(byteArr)
        Dim img As Bitmap = Image.FromStream(ms)
        Return img
    End Function

    Sub WriteImage(filename As String, bmp As Bitmap)
        bmp.Save(filename, bmp.RawFormat)
        bmp.Dispose()
    End Sub

    Function GenerateGaussianKernel(size As Integer, sigma As Single) As Single()
        Dim kernel(size * size - 1) As Single
        Dim mean As Integer = size \ 2
        Dim sum As Single = 0.0

        For x As Integer = 0 To size - 1
            For y As Integer = 0 To size - 1
                Dim xDistance As Integer = x - mean
                Dim yDistance As Integer = y - mean
                kernel(x * size + y) = Math.Exp(-((xDistance * xDistance + yDistance * yDistance) / (2 * sigma * sigma)))
                sum += kernel(x * size + y)
            Next
        Next

        ' Normalize the kernel to make the sum of all elements equal to 1
        For x As Integer = 0 To size - 1
            For y As Integer = 0 To size - 1
                kernel(x * size + y) /= sum
            Next
        Next

        Return kernel
    End Function

End Module