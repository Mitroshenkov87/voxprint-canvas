// Setup helper: checks that a downloaded file carries a valid Authenticode
// signature whose signer is the expected publisher.
// Usage: vxsigcheck.exe <file> "<publisher>"
// Exit 0 valid and expected publisher, 1 not trusted, 2 wrong publisher, 3 usage or error.
// Built by the Setup compiler with the .NET Framework csc.exe (C# 5).
using System;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;

static class VxSigCheck
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct WintrustFileInfo
    {
        public uint cbStruct;
        public string pcwszFilePath;
        public IntPtr hFile;
        public IntPtr pgKnownSubject;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct WintrustData
    {
        public uint cbStruct;
        public IntPtr pPolicyCallbackData;
        public IntPtr pSIPClientData;
        public uint dwUIChoice;
        public uint fdwRevocationChecks;
        public uint dwUnionChoice;
        public IntPtr pFile;
        public uint dwStateAction;
        public IntPtr hWVTStateData;
        public IntPtr pwszURLReference;
        public uint dwProvFlags;
        public uint dwUIContext;
        public IntPtr pSignatureSettings;
    }

    const uint WTD_UI_NONE = 2;
    const uint WTD_REVOKE_WHOLECHAIN = 1;
    const uint WTD_CHOICE_FILE = 1;
    const uint WTD_STATEACTION_VERIFY = 1;
    const uint WTD_STATEACTION_CLOSE = 2;
    const uint WTD_REVOCATION_CHECK_CHAIN_EXCLUDE_ROOT = 0x80;
    static readonly Guid GenericVerifyV2 = new Guid("00AAC56B-CD44-11d0-8CC2-00C04FC295EE");

    [DllImport("wintrust.dll", ExactSpelling = true, SetLastError = false)]
    static extern int WinVerifyTrust(IntPtr hwnd, [MarshalAs(UnmanagedType.LPStruct)] Guid action, ref WintrustData data);

    [DllImport("wintrust.dll", ExactSpelling = true)]
    static extern IntPtr WTHelperProvDataFromStateData(IntPtr hStateData);

    [DllImport("wintrust.dll", ExactSpelling = true)]
    static extern IntPtr WTHelperGetProvSignerFromChain(IntPtr pProvData, uint idxSigner, bool fCounterSigner, uint idxCounterSigner);

    [DllImport("wintrust.dll", ExactSpelling = true)]
    static extern IntPtr WTHelperGetProvCertFromChain(IntPtr pSgnr, uint idxCert);

    static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: vxsigcheck.exe <file> <publisher>");
            return 3;
        }
        string path = args[0];
        string publisher = args[1];
        WintrustFileInfo file = new WintrustFileInfo();
        file.cbStruct = (uint)Marshal.SizeOf(typeof(WintrustFileInfo));
        file.pcwszFilePath = path;
        IntPtr pFile = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(WintrustFileInfo)));
        Marshal.StructureToPtr(file, pFile, false);
        WintrustData data = new WintrustData();
        data.cbStruct = (uint)Marshal.SizeOf(typeof(WintrustData));
        data.dwUIChoice = WTD_UI_NONE;
        data.fdwRevocationChecks = WTD_REVOKE_WHOLECHAIN;
        data.dwUnionChoice = WTD_CHOICE_FILE;
        data.pFile = pFile;
        data.dwStateAction = WTD_STATEACTION_VERIFY;
        data.dwProvFlags = WTD_REVOCATION_CHECK_CHAIN_EXCLUDE_ROOT;
        try
        {
            int status = WinVerifyTrust(new IntPtr(-1), GenericVerifyV2, ref data);
            if (status != 0)
            {
                Console.Error.WriteLine("signature not trusted: 0x" + status.ToString("X8"));
                return 1;
            }
            IntPtr provData = WTHelperProvDataFromStateData(data.hWVTStateData);
            IntPtr signer = provData == IntPtr.Zero ? IntPtr.Zero : WTHelperGetProvSignerFromChain(provData, 0, false, 0);
            IntPtr provCert = signer == IntPtr.Zero ? IntPtr.Zero : WTHelperGetProvCertFromChain(signer, 0);
            if (provCert == IntPtr.Zero)
            {
                Console.Error.WriteLine("no signer certificate");
                return 1;
            }
            // CRYPT_PROVIDER_CERT: DWORD cbStruct, then PCCERT_CONTEXT pCert (pointer aligned).
            IntPtr certContext = Marshal.ReadIntPtr(provCert, IntPtr.Size);
            X509Certificate2 cert = new X509Certificate2(certContext);
            string name = cert.GetNameInfo(X509NameType.SimpleName, false);
            string org = null;
            foreach (string part in cert.Subject.Split(','))
            {
                string item = part.Trim();
                if (item.StartsWith("O=", StringComparison.Ordinal)) { org = item.Substring(2).Trim('"'); }
            }
            Console.WriteLine("signer: " + cert.Subject);
            if (!string.Equals(name, publisher, StringComparison.Ordinal) || !string.Equals(org, publisher, StringComparison.Ordinal))
            {
                Console.Error.WriteLine("unexpected signer");
                return 2;
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 3;
        }
        finally
        {
            data.dwStateAction = WTD_STATEACTION_CLOSE;
            WinVerifyTrust(new IntPtr(-1), GenericVerifyV2, ref data);
            Marshal.DestroyStructure(pFile, typeof(WintrustFileInfo));
            Marshal.FreeHGlobal(pFile);
        }
    }
}
