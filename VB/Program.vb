Imports System.Drawing
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

    <DllImport(dllFile, EntryPoint:="toCPU")>
    Public Function toCPU(ByVal gpuPtr As IntPtr, ByVal cpuPtr As Byte(), ByVal size As Long) As Boolean
    End Function

    <DllImport(dllFile, EntryPoint:="conv2DCuda")>
    Public Sub convolution(ByVal src As IntPtr, ByVal dst As IntPtr, ByVal channels As Integer, ByVal width As Integer, ByVal height As Integer, ByVal kernel As Single(), ByVal kernelSize As Integer)
    End Sub

    Sub Main(args As String())
        Console.WriteLine("CUDA device count: " & deviceCount())

        Dim img As Bitmap = OpenImage("../../../image/lena_color.bmp")
        Dim channels As Integer = 3

        ' bitmap to byte array
        Dim bmpData As Imaging.BitmapData = img.LockBits(New Rectangle(0, 0, img.Width, img.Height), Imaging.ImageLockMode.ReadWrite, img.PixelFormat)
        Dim ptr As IntPtr = bmpData.Scan0
        Dim bytes As Integer = Math.Abs(bmpData.Stride) * img.Height
        Dim src(bytes - 1) As Byte
        Marshal.Copy(ptr, src, 0, bytes)
        img.UnlockBits(bmpData)

        ' 宣告指向 GPU 記憶體的指標
        Dim devSrc As IntPtr = Nothing
        Dim devDst As IntPtr = Nothing

        ' 分配 GPU 記憶體
        cudaMalloc(devSrc, bytes)
        cudaMalloc(devDst, bytes)

        ' byte array 複製到 GPU 記憶體中
        toGPU(src, devSrc, bytes)

        ' 呼叫 CUDA 函式 (src, dst 都是指向 GPU 記憶體的指標，只要資料還在 GPU 中就可以重複 call CUDA 函式)
        Dim kernel As Single() = GenerateGaussianKernel(5, 1.0)
        convolution(devSrc, devDst, channels, img.Width, img.Height, kernel, 5)
        convolution(devDst, devSrc, channels, img.Width, img.Height, kernel, 5)
        convolution(devSrc, devDst, channels, img.Width, img.Height, kernel, 5)
        convolution(devDst, devSrc, channels, img.Width, img.Height, kernel, 5)
        convolution(devSrc, devDst, channels, img.Width, img.Height, kernel, 5)

        ' 從 GPU 記憶體中複製資料回來
        Dim dst(bytes - 1) As Byte
        toCPU(devDst, dst, bytes)

        ' 複製 byte array 到新的 bitmap
        Dim img2 As New Bitmap(img.Width, img.Height, img.PixelFormat)
        Dim bmpData2 As Imaging.BitmapData = img2.LockBits(New Rectangle(0, 0, img.Width, img.Height), Imaging.ImageLockMode.ReadWrite, img2.PixelFormat)
        Dim ptr2 As IntPtr = bmpData2.Scan0
        Marshal.Copy(dst, 0, ptr2, bytes)
        img2.UnlockBits(bmpData2)

        WriteImage("../../../image/lena_new.bmp", img2)

        ' 釋放 GPU 記憶體
        cudaFree(devSrc)
        cudaFree(devDst)
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